#pragma once

#include <string>
#include <vector>

#include <flexon/online/runtime_engine.hpp>

namespace flexon::online::cli {

std::vector<core::Resource> parse_resource_plan(const std::string& value);
core::Resource parse_resource(const std::string& value);

void print_usage(const char* program);

}  // namespace flexon::online::cli
