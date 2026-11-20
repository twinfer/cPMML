/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_TYPES_STRING_POOL_H
#define CPMML_TYPES_STRING_POOL_H

/**
 * Design note -- strings: text is identity, interning is a cache (§2.3).
 *
 * XSD string equality is equality of text, so two Strings are equal exactly
 * when their texts are (string_equal, value.h). Interning only makes the
 * common case cheap:
 *
 * - StringPool: ONE per document, shared by every model and segment. Every
 *   string the document names is interned at load; after freeze() the pool
 *   is read-only, and find() may be called from any number of threads. Two
 *   reps of one pool are equal iff they are the same pointer. Interning
 *   after freeze() is a programming error and throws (no silent fallback).
 * - RecordArena: per evaluation. Strings made while scoring (an input the
 *   pool never saw, uppercase(), concat(), a cast to string) get a rep here,
 *   with id kUnpooled, and are compared by text.
 *
 * A rep from one document's pool must never meet a rep from another's: both
 * have ids, so string_equal would call them different without reading text.
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory_resource>
#include <new>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

#include "core/types/value.h"

namespace cpmml::types {

namespace detail {
inline const StringRep* make_rep(std::pmr::memory_resource& arena, std::string_view text, std::uint32_t id) {
  char* bytes = static_cast<char*>(arena.allocate(text.size() + 1, 1));
  if (!text.empty()) std::memcpy(bytes, text.data(), text.size());
  bytes[text.size()] = '\0';
  void* where = arena.allocate(sizeof(StringRep), alignof(StringRep));
  return ::new (where) StringRep{std::string_view(bytes, text.size()), id, std::hash<std::string_view>{}(text)};
}
}  // namespace detail

class StringPool {
 public:
  StringPool() = default;
  StringPool(const StringPool&) = delete;
  StringPool& operator=(const StringPool&) = delete;

  // Load only: the pooled rep of `text`, made on first use.
  const StringRep* intern(std::string_view text) {
    if (frozen_) throw std::logic_error("StringPool::intern after freeze: the document's pool is read-only");
    if (const auto it = index_.find(text); it != index_.end()) return it->second;
    if (next_id_ == StringRep::kUnpooled) throw std::length_error("StringPool: too many distinct strings");
    const StringRep* rep = detail::make_rep(arena_, text, next_id_++);
    index_.emplace(rep->text, rep);  // the key views the rep's own bytes
    return rep;
  }

  // The pooled rep of `text`, or nullptr. Read-only; thread-safe once frozen.
  const StringRep* find(std::string_view text) const {
    const auto it = index_.find(text);
    return it == index_.end() ? nullptr : it->second;
  }

  void freeze() { frozen_ = true; }
  bool frozen() const { return frozen_; }
  std::size_t size() const { return index_.size(); }

 private:
  std::pmr::monotonic_buffer_resource arena_;
  std::unordered_map<std::string_view, const StringRep*> index_;
  std::uint32_t next_id_ = 0;
  bool frozen_ = false;
};

class RecordArena {
 public:
  RecordArena() = default;
  // Reps go to `buffer` first; reset() keeps it, so an arena reused per
  // evaluation allocates nothing while a record's strings fit there.
  RecordArena(void* buffer, std::size_t size) : arena_(buffer, size) {}
  RecordArena(const RecordArena&) = delete;
  RecordArena& operator=(const RecordArena&) = delete;

  const StringRep* make(std::string_view text) { return detail::make_rep(arena_, text, StringRep::kUnpooled); }
  // Every rep made so far dies here.
  void reset() { arena_.release(); }

 private:
  std::pmr::monotonic_buffer_resource arena_;
};

}  // namespace cpmml::types

#endif  // CPMML_TYPES_STRING_POOL_H
