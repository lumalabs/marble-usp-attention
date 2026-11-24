// SPDX-FileCopyrightText: 2026 Luma AI, Inc.
// SPDX-License-Identifier: MIT
#pragma once
// Shared across heads/batches. Metadata rows are segment, noise, document, position.
struct SparseArgs {
  const int* counts = nullptr;
  const int* indices = nullptr;
  const int* types = nullptr;
  const int* q_aux = nullptr;
  const int* kv_aux = nullptr;
  const int* mask_offsets = nullptr;
  const int* mask_bits = nullptr;
  const int* q_order = nullptr;
};
