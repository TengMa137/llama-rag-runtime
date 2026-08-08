#include "rag/text/chunker.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <sstream>

namespace rag::text {
namespace {

struct Span {
    std::string text;
    std::uint32_t line = 0;
};

std::size_t count(std::string_view text, const ChunkOptions& opts) {
    return opts.measure_tokens ? opts.measure_tokens(text) : text.size();
}

std::size_t next_utf8(std::string_view text, std::size_t at) {
    if (at >= text.size())
        return text.size();
    const unsigned char c = static_cast<unsigned char>(text[at]);
    std::size_t width = c < 0x80             ? 1
                        : (c & 0xe0) == 0xc0 ? 2
                        : (c & 0xf0) == 0xe0 ? 3
                        : (c & 0xf8) == 0xf0 ? 4
                                             : 1;
    if (at + width > text.size())
        return at + 1;
    for (std::size_t i = 1; i < width; ++i)
        if ((static_cast<unsigned char>(text[at + i]) & 0xc0) != 0x80)
            return at + 1;
    return at + width;
}

bool sentence_boundary(char c) { return c == '.' || c == '!' || c == '?' || c == ';'; }
bool soft_boundary(char c) {
    return std::isspace(static_cast<unsigned char>(c)) || c == ',' || c == ':' || c == '/' ||
           c == '}' || c == ']' || c == ')' || c == '-' || c == '|';
}

std::size_t usable_limit(const ChunkOptions& opts) {
    if (opts.policy.max_tokens == 0)
        return std::max<std::size_t>(1, opts.max_chars);
    if (opts.policy.reserved_tokens >= opts.policy.max_tokens)
        return 1;
    return opts.policy.max_tokens - opts.policy.reserved_tokens;
}

std::string embedding_form(std::string_view context, std::string_view body,
                           const ChunkOptions& opts) {
    std::string out = opts.policy.document_prefix;
    if (!out.empty() && (!context.empty() || !body.empty()))
        out.push_back('\n');
    if (!context.empty()) {
        out.append(context);
        if (!body.empty())
            out.push_back('\n');
    }
    out.append(body);
    return out;
}

// Choose a UTF-8 boundary that fits the hard limit, preferring sentence and
// then whitespace/code/list boundaries. Always consumes at least one scalar.
std::size_t split_at(std::string_view text, std::size_t begin, std::string_view context,
                     const ChunkOptions& opts) {
    const std::size_t limit = usable_limit(opts);
    std::size_t at = begin;
    std::size_t last_fit = begin;
    std::size_t sentence = begin;
    std::size_t soft = begin;
    while (at < text.size()) {
        const std::size_t end = next_utf8(text, at);
        if (count(embedding_form(context, text.substr(begin, end - begin), opts), opts) > limit)
            break;
        last_fit = end;
        const char tail = text[end - 1];
        if (sentence_boundary(tail))
            sentence = end;
        if (soft_boundary(tail))
            soft = end;
        at = end;
    }
    if (last_fit == text.size())
        return last_fit;
    if (sentence > begin)
        return sentence;
    if (soft > begin)
        return soft;
    return last_fit > begin ? last_fit : next_utf8(text, begin);
}

int heading_level(std::string_view line) {
    std::size_t i = 0;
    while (i < line.size() && line[i] == '#')
        ++i;
    return i >= 1 && i <= 6 && i < line.size() && line[i] == ' ' ? static_cast<int>(i) : 0;
}

std::string heading_text(std::string_view line) {
    std::size_t i = 0;
    while (i < line.size() && (line[i] == '#' || line[i] == ' '))
        ++i;
    std::string out(line.substr(i));
    while (!out.empty() && (out.back() == ' ' || out.back() == '#'))
        out.pop_back();
    return out;
}

std::string normalize_input(std::string_view body, const ChunkOptions& opts, bool& valid) {
    std::string out;
    out.reserve(body.size());
    for (std::size_t i = 0; i < body.size(); ++i) {
        if (body[i] == '\r') {
            if (i + 1 < body.size() && body[i + 1] == '\n')
                ++i;
            out.push_back('\n');
        } else if (static_cast<unsigned char>(body[i]) < 0x80) {
            out.push_back(body[i]);
        } else {
            const std::size_t end = next_utf8(body, i);
            if (end == i + 1) {
                if (opts.policy.invalid_utf8 == InvalidUtf8Policy::reject) {
                    valid = false;
                    return {};
                }
                out.append("\xef\xbf\xbd");
            } else {
                out.append(body.substr(i, end - i));
                i = end - 1;
            }
        }
    }
    return out;
}

} // namespace

std::string chunking_fingerprint(const ChunkOptions& o) {
    std::ostringstream s;
    // Only fields that can change structural boundaries belong here. Model
    // identity, vector dimension, and query prefix are validated separately by
    // coordinators and must not make desktop/mobile chunk geometry diverge.
    s << "rag.cpp-structural-v2|" << o.max_lines << '|' << o.max_chars << '|' << o.overlap_lines
      << '|' << o.heading_context << '|' << o.policy.tokenizer_identity << '|'
      << o.policy.target_tokens << '|' << o.policy.max_tokens << '|' << o.policy.overlap_tokens
      << '|' << o.policy.reserved_tokens << '|' << o.policy.document_prefix << '|'
      << static_cast<int>(o.policy.counting_mode) << '|' << static_cast<int>(o.policy.invalid_utf8);
    return s.str();
}

std::vector<Chunk> chunk_document(DocId doc_id, const std::string& input,
                                  const ChunkOptions& opts) {
    bool valid = true;
    const std::string body = normalize_input(input, opts, valid);
    if (!valid || body.empty())
        return {};
    if (count(opts.policy.document_prefix, opts) >= usable_limit(opts) &&
        !opts.policy.document_prefix.empty())
        return {};

    std::vector<Span> spans;
    std::uint32_t line_number = 0;
    for (std::size_t begin = 0; begin <= body.size();) {
        const std::size_t newline = body.find('\n', begin);
        const std::size_t end = newline == std::string::npos ? body.size() : newline;
        std::string_view line(body.data() + begin, end - begin);
        if (line.empty()) {
            spans.push_back({{}, line_number});
        } else {
            for (std::size_t at = 0; at < line.size();) {
                const std::size_t cut = split_at(line, at, {}, opts);
                spans.push_back({std::string(line.substr(at, cut - at)), line_number});
                at = cut;
            }
        }
        if (newline == std::string::npos)
            break;
        begin = newline + 1;
        ++line_number;
    }

    std::vector<Chunk> out;
    std::array<std::string, 7> headings{};
    const std::size_t target =
        opts.policy.target_tokens ? opts.policy.target_tokens : usable_limit(opts);
    std::size_t i = 0;
    while (i < spans.size()) {
        const std::size_t start = i;
        std::string text;
        std::string context;
        std::size_t lines = 0;
        while (i < spans.size() && lines < std::max<std::size_t>(1, opts.max_lines)) {
            Span span = spans[i];
            if (opts.heading_context) {
                if (int level = heading_level(span.text); level > 0) {
                    headings[level] = heading_text(span.text);
                    for (int deeper = level + 1; deeper <= 6; ++deeper)
                        headings[deeper].clear();
                }
                if (context.empty()) {
                    for (int level = 1; level <= 6; ++level) {
                        if (headings[level].empty())
                            continue;
                        if (!context.empty())
                            context += " › ";
                        context += headings[level];
                    }
                }
            }
            std::string candidate = text;
            if (!candidate.empty())
                candidate.push_back('\n');
            candidate += span.text;
            const auto tokens = count(embedding_form(context, candidate, opts), opts);
            if (!text.empty() && tokens > usable_limit(opts))
                break;
            if (text.empty() && tokens > usable_limit(opts)) {
                const std::size_t cut = split_at(span.text, 0, context, opts);
                if (cut < span.text.size()) {
                    spans[i].text = span.text.substr(0, cut);
                    spans.insert(spans.begin() + static_cast<std::ptrdiff_t>(i + 1),
                                 Span{span.text.substr(cut), span.line});
                    continue;
                }
                // A heading breadcrumb alone can consume a model's budget.
                // Drop it rather than violate the hard embedding limit.
                context.clear();
                candidate = span.text;
            }
            text = std::move(candidate);
            ++i;
            ++lines;
            if (tokens >= target || (span.text.empty() && tokens >= target / 2))
                break;
        }
        while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
            text.pop_back();
        if (!text.empty()) {
            Chunk chunk;
            chunk.doc = doc_id;
            chunk.text = std::move(text);
            chunk.context = std::move(context);
            chunk.start_line = spans[start].line;
            chunk.end_line = spans[i - 1].line;
            out.push_back(std::move(chunk));
        }
        if (i == start)
            ++i;

        if (i < spans.size()) {
            const std::size_t overlap =
                opts.policy.max_tokens ? opts.policy.overlap_tokens : opts.overlap_lines;
            std::size_t back = 0;
            std::size_t j = i;
            while (j > start + 1 && back < overlap) {
                --j;
                back += opts.policy.max_tokens ? count(spans[j].text, opts) : 1;
            }
            if (j > start && j < i)
                i = j;
        }
    }
    return out;
}

} // namespace rag::text
