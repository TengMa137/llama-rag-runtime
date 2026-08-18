#include "rag/ingestion/job_store.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include <fcntl.h>
#include <nlohmann/json.hpp>
#include <unistd.h>

#include "rag/store/format.hpp"

namespace rag::ingestion {
namespace {

using json = nlohmann::json;
constexpr std::uint32_t kJobMagic = 0x52424F4A; // "JOBR" little-endian
constexpr std::size_t kHeaderBytes = 12;

IngestionJob decode(std::string_view payload);

Result<std::size_t> valid_prefix(std::string_view blob) {
    std::size_t position = 0;
    std::size_t records = 0;
    while (position + kHeaderBytes <= blob.size()) {
        if (++records > store::kMaxWalRecords)
            return fail<std::size_t>(Errc::corrupt_index, "job record limit exceeded");
        store::Reader reader(blob.substr(position, kHeaderBytes));
        std::uint32_t magic = 0, length = 0, checksum = 0;
        if (!reader.u(magic) || !reader.u(length) || !reader.u(checksum))
            return position;
        if (magic != kJobMagic || length > store::kMaxWalRecordBytes)
            return fail<std::size_t>(Errc::corrupt_index, "job record header is invalid");
        const std::size_t body = position + kHeaderBytes;
        if (length > blob.size() - body)
            return position;
        const std::string_view payload = blob.substr(body, length);
        if (store::crc32(payload) != checksum) {
            if (body + length == blob.size())
                return position;
            return fail<std::size_t>(Errc::corrupt_index, "job record checksum mismatch");
        }
        try {
            (void)decode(payload);
        } catch (...) {
            if (body + length == blob.size())
                return position;
            return fail<std::size_t>(Errc::corrupt_index, "job record payload is invalid");
        }
        position = body + length;
    }
    return position;
}

Result<std::string> read_log(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return std::string{};
    std::stringstream stream;
    stream << input.rdbuf();
    if (!input.good() && !input.eof())
        return fail<std::string>(Errc::io_error, "job store read failed");
    return stream.str();
}

json prepared_json(const backend::PreparedDocument& document) {
    json chunks = json::array();
    for (const auto& chunk : document.chunks)
        chunks.push_back({{"key", chunk.key},
                          {"ordinal", chunk.ordinal},
                          {"text", chunk.text},
                          {"indexed_text", chunk.indexed_text},
                          {"context", chunk.context},
                          {"start_line", chunk.start_line},
                          {"end_line", chunk.end_line},
                          {"embedding", chunk.embedding}});
    return {{"key", document.key},
            {"title", document.title},
            {"content", document.content},
            {"metadata", document.metadata},
            {"content_hash", document.content_hash},
            {"chunking_fingerprint", document.chunking_fingerprint},
            {"embedding_identity", document.embedding_identity},
            {"chunks", std::move(chunks)}};
}

backend::PreparedDocument parse_prepared(const json& value) {
    backend::PreparedDocument document;
    document.key = value.at("key").get<std::string>();
    document.title = value.at("title").get<std::string>();
    document.content = value.at("content").get<std::string>();
    document.metadata = value.at("metadata").get<Metadata>();
    document.content_hash = value.at("content_hash").get<std::string>();
    document.chunking_fingerprint = value.at("chunking_fingerprint").get<std::string>();
    document.embedding_identity = value.value("embedding_identity", "");
    const auto& chunks = value.at("chunks");
    if (!chunks.is_array() || chunks.size() > store::kMaxChunks)
        throw std::runtime_error("job prepared chunk count is invalid");
    document.chunks.reserve(chunks.size());
    for (const auto& value_chunk : chunks) {
        backend::PreparedChunk chunk;
        chunk.key = value_chunk.at("key").get<std::string>();
        chunk.ordinal = value_chunk.at("ordinal").get<std::size_t>();
        chunk.text = value_chunk.at("text").get<std::string>();
        chunk.indexed_text = value_chunk.at("indexed_text").get<std::string>();
        chunk.context = value_chunk.at("context").get<std::string>();
        chunk.start_line = value_chunk.at("start_line").get<std::uint32_t>();
        chunk.end_line = value_chunk.at("end_line").get<std::uint32_t>();
        chunk.embedding = value_chunk.at("embedding").get<Vector>();
        if (chunk.embedding.size() > store::kMaxVectorDimension)
            throw std::runtime_error("job embedding dimension is invalid");
        document.chunks.push_back(std::move(chunk));
    }
    return document;
}

std::string encode(const IngestionJob& job) {
    json value = {{"version", 1},
                  {"id", job.id},
                  {"operation", static_cast<int>(job.operation)},
                  {"document", job.input.document},
                  {"title", job.input.title},
                  {"content", job.input.content},
                  {"metadata", job.input.metadata},
                  {"revision", job.revision},
                  {"content_hash", job.content_hash},
                  {"status", static_cast<int>(job.status)},
                  {"created_at_ms", job.created_at_ms},
                  {"updated_at_ms", job.updated_at_ms}};
    if (job.error)
        value["error"] = {{"code", static_cast<int>(job.error->code)},
                          {"message", job.error->message}};
    if (job.prepared)
        value["prepared"] = prepared_json(*job.prepared);
    if (job.mutation_applied)
        value["mutation_applied"] = *job.mutation_applied;
    return value.dump();
}

Result<std::string> encode_frame(const IngestionJob& job) {
    const std::string payload = encode(job);
    if (payload.size() > store::kMaxWalRecordBytes || payload.size() > UINT32_MAX)
        return fail<std::string>(Errc::invalid_argument, "job record exceeds configured limit");
    store::Writer writer;
    writer.u<std::uint32_t>(kJobMagic);
    writer.u<std::uint32_t>(static_cast<std::uint32_t>(payload.size()));
    writer.u<std::uint32_t>(store::crc32(payload));
    writer.bytes(payload);
    return std::move(writer.data());
}

IngestionJob decode(std::string_view payload) {
    const auto value = json::parse(payload);
    if (value.at("version") != 1)
        throw std::runtime_error("unsupported job record version");
    IngestionJob job;
    job.id = value.at("id").get<std::string>();
    const int operation = value.value("operation", static_cast<int>(JobOperation::upsert));
    if (operation < static_cast<int>(JobOperation::upsert) ||
        operation > static_cast<int>(JobOperation::erase))
        throw std::runtime_error("job operation is invalid");
    job.operation = static_cast<JobOperation>(operation);
    job.input.document = value.at("document").get<std::string>();
    job.input.title = value.at("title").get<std::string>();
    job.input.content = value.at("content").get<std::string>();
    job.input.metadata = value.at("metadata").get<Metadata>();
    job.revision = value.at("revision").get<backend::DocumentRevision>();
    job.content_hash = value.at("content_hash").get<std::string>();
    const int status = value.at("status").get<int>();
    if (status < static_cast<int>(JobStatus::queued) ||
        status > static_cast<int>(JobStatus::cancelled))
        throw std::runtime_error("job status is invalid");
    job.status = static_cast<JobStatus>(status);
    job.created_at_ms = value.at("created_at_ms").get<std::int64_t>();
    job.updated_at_ms = value.at("updated_at_ms").get<std::int64_t>();
    if (value.contains("error")) {
        const int error_code = value.at("error").at("code").get<int>();
        if (error_code < static_cast<int>(Errc::ok) ||
            error_code > static_cast<int>(Errc::corrupt_index))
            throw std::runtime_error("job error code is invalid");
        job.error = JobError{static_cast<Errc>(error_code),
                             value.at("error").at("message").get<std::string>()};
    }
    if (value.contains("prepared"))
        job.prepared = parse_prepared(value.at("prepared"));
    if (value.contains("mutation_applied"))
        job.mutation_applied = value.at("mutation_applied").get<bool>();
    if (job.id.empty() || job.input.document.empty() || job.revision == 0)
        throw std::runtime_error("job identity is invalid");
    return job;
}

Result<std::vector<IngestionJob>> latest_jobs(std::string_view blob) {
    try {
        std::unordered_map<JobId, IngestionJob> latest;
        std::size_t position = 0;
        std::size_t records = 0;
        while (position + kHeaderBytes <= blob.size()) {
            if (++records > store::kMaxWalRecords)
                return fail<std::vector<IngestionJob>>(Errc::corrupt_index,
                                                       "job record limit exceeded");
            store::Reader reader(blob.substr(position, kHeaderBytes));
            std::uint32_t magic = 0, length = 0, checksum = 0;
            if (!reader.u(magic) || !reader.u(length) || !reader.u(checksum))
                break;
            if (magic != kJobMagic || length > store::kMaxWalRecordBytes)
                return fail<std::vector<IngestionJob>>(Errc::corrupt_index,
                                                       "job record header is invalid");
            const std::size_t body = position + kHeaderBytes;
            if (length > blob.size() - body)
                break;
            const std::string_view payload = blob.substr(body, length);
            if (store::crc32(payload) != checksum) {
                if (body + length == blob.size())
                    break;
                return fail<std::vector<IngestionJob>>(Errc::corrupt_index,
                                                       "job record checksum mismatch");
            }
            auto job = decode(payload);
            latest[job.id] = std::move(job);
            position = body + length;
        }
        std::vector<IngestionJob> output;
        output.reserve(latest.size());
        for (auto& [id, job] : latest)
            output.push_back(std::move(job));
        std::sort(output.begin(), output.end(), [](const auto& left, const auto& right) {
            if (left.created_at_ms != right.created_at_ms)
                return left.created_at_ms < right.created_at_ms;
            return left.id < right.id;
        });
        return output;
    } catch (const std::exception& error) {
        return fail<std::vector<IngestionJob>>(Errc::corrupt_index,
                                               "job replay failed: " + std::string(error.what()));
    }
}

} // namespace

Result<std::string> serialize_ingestion_job(const IngestionJob& job) {
    try {
        const std::string payload = encode(job);
        if (payload.size() > store::kMaxWalRecordBytes)
            return fail<std::string>(Errc::invalid_argument, "job record exceeds configured limit");
        return payload;
    } catch (const std::exception& error) {
        return fail<std::string>(Errc::invalid_argument,
                                 "job record serialization failed: " + std::string(error.what()));
    }
}

Result<IngestionJob> deserialize_ingestion_job(std::string_view payload) {
    try {
        return decode(payload);
    } catch (const std::exception& error) {
        return fail<IngestionJob>(Errc::corrupt_index,
                                  "job record is invalid: " + std::string(error.what()));
    }
}

Result<std::vector<IngestionJob>> load_ingestion_jobs_read_only(const std::string& path) {
    auto blob = read_log(path);
    if (!blob)
        return unexpected(blob.error());
    return latest_jobs(*blob);
}

AppendOnlyJobStore::~AppendOnlyJobStore() {
    std::lock_guard lock(mutex_);
    if (fd_ >= 0)
        ::close(fd_);
}

Result<std::shared_ptr<AppendOnlyJobStore>> AppendOnlyJobStore::open(std::string path,
                                                                     store::SyncMode mode) {
    auto store = std::make_shared<AppendOnlyJobStore>();
    if (auto opened = store->open_file(std::move(path), mode); !opened)
        return unexpected(opened.error());
    return store;
}

Result<void> AppendOnlyJobStore::open_file(std::string path, store::SyncMode mode) {
    std::lock_guard lock(mutex_);
    auto blob = read_log(path);
    if (!blob)
        return unexpected(blob.error());
    auto intact = valid_prefix(*blob);
    if (!intact)
        return unexpected(intact.error());
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0600);
    if (fd < 0)
        return fail<void>(Errc::io_error, "job store open " + path + ": " + std::strerror(errno));
    fd_ = fd;
    path_ = std::move(path);
    mode_ = mode;
    if (*intact < blob->size()) {
        if (::ftruncate(fd_, static_cast<off_t>(*intact)) != 0) {
            ::close(fd_);
            fd_ = -1;
            return fail<void>(Errc::io_error,
                              "job store tail repair failed: " + std::string(std::strerror(errno)));
        }
        if (auto synced = sync_now(); !synced) {
            ::close(fd_);
            fd_ = -1;
            return synced;
        }
    }
    ::lseek(fd_, 0, SEEK_END);
    bytes_ = *intact;
    return {};
}

Result<void> AppendOnlyJobStore::sync_now() noexcept {
    if (mode_ == store::SyncMode::none)
        return {};
#if defined(F_FULLFSYNC)
    if (mode_ == store::SyncMode::full && ::fcntl(fd_, F_FULLFSYNC) == 0)
        return {};
#endif
    if (::fsync(fd_) != 0)
        return fail<void>(Errc::io_error, "job store fsync failed");
    return {};
}

Result<void> AppendOnlyJobStore::persist(const IngestionJob& job) {
    try {
        auto encoded = encode_frame(job);
        if (!encoded)
            return unexpected(encoded.error());
        const std::string& frame = *encoded;

        std::lock_guard lock(mutex_);
        if (fd_ < 0)
            return fail<void>(Errc::io_error, "job store is not open");
        std::size_t offset = 0;
        while (offset < frame.size()) {
            const ssize_t written = ::write(fd_, frame.data() + offset, frame.size() - offset);
            if (written < 0) {
                if (errno == EINTR)
                    continue;
                return fail<void>(Errc::io_error,
                                  "job store write: " + std::string(std::strerror(errno)));
            }
            offset += static_cast<std::size_t>(written);
        }
        if (auto synced = sync_now(); !synced)
            return synced;
        bytes_ += frame.size();
        return {};
    } catch (const std::exception& error) {
        return fail<void>(Errc::io_error, "job store append failed: " + std::string(error.what()));
    }
}

Result<std::vector<IngestionJob>> AppendOnlyJobStore::load_latest() const {
    std::lock_guard lock(mutex_);
    return load_ingestion_jobs_read_only(path_);
}

std::uint64_t AppendOnlyJobStore::size_bytes() const noexcept {
    std::lock_guard lock(mutex_);
    return bytes_;
}

Result<void> AppendOnlyJobStore::truncate_prefix(std::uint64_t represented_position,
                                                 JobRetentionPolicy retention) {
    try {
        std::lock_guard lock(mutex_);
        if (fd_ < 0)
            return fail<void>(Errc::io_error, "job store is not open");
        auto blob = read_log(path_);
        if (!blob)
            return unexpected(blob.error());
        auto intact = valid_prefix(*blob);
        if (!intact)
            return unexpected(intact.error());
        if (represented_position > *intact)
            return fail<void>(Errc::invalid_argument,
                              "checkpoint position exceeds complete job log");
        const auto prefix_size = static_cast<std::size_t>(represented_position);
        auto prefix = valid_prefix(std::string_view(*blob).substr(0, prefix_size));
        if (!prefix || *prefix != prefix_size)
            return fail<void>(Errc::invalid_argument,
                              "checkpoint position is not a job-frame boundary");
        if (represented_position == 0 && *intact == blob->size())
            return {};

        std::unordered_map<JobId, IngestionJob> latest;
        std::size_t position = 0;
        while (position < prefix_size) {
            store::Reader reader(std::string_view(*blob).substr(position, kHeaderBytes));
            std::uint32_t magic = 0, length = 0, checksum = 0;
            if (!reader.u(magic) || !reader.u(length) || !reader.u(checksum))
                return fail<void>(Errc::corrupt_index, "job prefix changed during compaction");
            const std::size_t body = position + kHeaderBytes;
            auto job = decode(std::string_view(*blob).substr(body, length));
            latest[job.id] = std::move(job);
            position = body + length;
        }
        std::vector<IngestionJob> selected;
        std::vector<IngestionJob> terminal_jobs;
        selected.reserve(latest.size());
        terminal_jobs.reserve(latest.size());
        for (auto& [id, job] : latest) {
            if (terminal(job.status))
                terminal_jobs.push_back(std::move(job));
            else
                selected.push_back(std::move(job));
        }
        std::sort(terminal_jobs.begin(), terminal_jobs.end(),
                  [](const auto& left, const auto& right) {
                      if (left.updated_at_ms != right.updated_at_ms)
                          return left.updated_at_ms > right.updated_at_ms;
                      return left.id > right.id;
                  });
        std::size_t retained_terminal_bytes = 0;
        std::size_t retained_terminal_jobs = 0;
        for (auto& job : terminal_jobs) {
            if (retained_terminal_jobs >= retention.max_terminal_jobs)
                break;
            auto encoded = encode_frame(job);
            if (!encoded)
                return unexpected(encoded.error());
            if (encoded->size() > retention.max_terminal_bytes - retained_terminal_bytes)
                continue;
            retained_terminal_bytes += encoded->size();
            ++retained_terminal_jobs;
            selected.push_back(std::move(job));
        }
        std::sort(selected.begin(), selected.end(), [](const auto& left, const auto& right) {
            if (left.created_at_ms != right.created_at_ms)
                return left.created_at_ms < right.created_at_ms;
            return left.id < right.id;
        });
        std::string compacted;
        for (const auto& job : selected) {
            auto encoded = encode_frame(job);
            if (!encoded)
                return unexpected(encoded.error());
            compacted += *encoded;
        }
        const std::string_view tail(blob->data() + prefix_size, *intact - prefix_size);
        compacted.append(tail.data(), tail.size());
        static std::atomic<std::uint64_t> sequence{0};
        const std::string temporary = path_ + ".compact." + std::to_string(::getpid()) + "." +
                                      std::to_string(sequence.fetch_add(1));
        const int temporary_fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (temporary_fd < 0)
            return fail<void>(Errc::io_error, "job store compact open failed: " +
                                                  std::string(std::strerror(errno)));
        std::size_t written = 0;
        while (written < compacted.size()) {
            const ssize_t count =
                ::write(temporary_fd, compacted.data() + written, compacted.size() - written);
            if (count < 0) {
                if (errno == EINTR)
                    continue;
                ::close(temporary_fd);
                ::unlink(temporary.c_str());
                return fail<void>(Errc::io_error, "job store compact write failed: " +
                                                      std::string(std::strerror(errno)));
            }
            written += static_cast<std::size_t>(count);
        }
        bool synchronized = mode_ == store::SyncMode::none;
#if defined(F_FULLFSYNC)
        if (mode_ == store::SyncMode::full && ::fcntl(temporary_fd, F_FULLFSYNC) == 0)
            synchronized = true;
#endif
        if (!synchronized && ::fsync(temporary_fd) == 0)
            synchronized = true;
        if (!synchronized) {
            ::close(temporary_fd);
            ::unlink(temporary.c_str());
            return fail<void>(Errc::io_error, "job store compact fsync failed");
        }
        if (::close(temporary_fd) != 0) {
            ::unlink(temporary.c_str());
            return fail<void>(Errc::io_error, "job store compact close failed");
        }
        if (::rename(temporary.c_str(), path_.c_str()) != 0) {
            ::unlink(temporary.c_str());
            return fail<void>(Errc::io_error, "job store compact rename failed: " +
                                                  std::string(std::strerror(errno)));
        }
        const auto slash = path_.find_last_of('/');
        const std::string directory =
            slash == std::string::npos ? std::string(".") : path_.substr(0, slash);
        if (const int directory_fd = ::open(directory.c_str(), O_RDONLY); directory_fd >= 0) {
            (void)::fsync(directory_fd);
            ::close(directory_fd);
        }
        ::close(fd_);
        fd_ = ::open(path_.c_str(), O_WRONLY | O_APPEND);
        if (fd_ < 0)
            return fail<void>(Errc::io_error, "job store reopen after compact failed");
        bytes_ = compacted.size();
        return {};
    } catch (const std::exception& error) {
        return fail<void>(Errc::io_error,
                          "job store prefix compact failed: " + std::string(error.what()));
    }
}

} // namespace rag::ingestion
