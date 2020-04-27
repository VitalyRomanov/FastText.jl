Lessons

- Prefer column wise indexing
- `for i = 1:100` is slower than while loop because it allocates memory
- any slice, even @views will allocate memory
- @inbounds does not help much in performance
- apparently accessing fields of mutable (or even immutable structs) is slow