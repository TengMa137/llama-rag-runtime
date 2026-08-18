#pragma once

#include <memory>
#include <string>

#include "rag/migration/contract.hpp"

namespace rag::migration {

// Source mode loads the checkpoint plus ready records from its sibling .jobs
// log without opening either file for write. Destination mode requires a new
// path or a matching resumable migration sidecar.
[[nodiscard]] Result<std::unique_ptr<Endpoint>> open_embedded_source(std::string checkpoint_path);
[[nodiscard]] Result<std::unique_ptr<Endpoint>>
open_embedded_destination(std::string checkpoint_path);

} // namespace rag::migration
