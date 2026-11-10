# Padding of the reconstruction grid (`padding` keyword of `deconvolution`, `map_cj`,
# `map_cj_lean`, `map_cj_exact`): one fraction for all dimensions (as before), or one per
# dimension as a tuple or vector, e.g. `(0.05, 0.05, 0.25)` for a 3D wide-field stack, whose
# out-of-focus light reaches much further along z than laterally.
#
# Per dimension d with fraction p = padding[d] (and size n > 1), the grid grows by
# `next_fast_fft_size(max(4, 2 round(n |p|)))` voxels (split over both sides); p = 0 does not pad.
# The result is cropped back to the measured size in dimensions with p > 0 and keeps the padded
# size where p < 0.

_valid_padding(p::Real, N) = isfinite(p)
_valid_padding(p::Union{Tuple,AbstractVector}, N) = length(p) == N && all(v -> v isa Real && isfinite(v), p)
_valid_padding(p, N) = false

function _check_padding(p, N)
    _valid_padding(p, N) ||
        throw(ArgumentError("`padding` must be a finite number or one finite number per dimension ($N), got $p"))
    return nothing
end

_padding_fraction(p::Real, d) = p
_padding_fraction(p, d) = p[d]

"padded grid size for data of size `sz`"
_padded_size(sz::Tuple, padding) = ntuple(length(sz)) do d
    p = _padding_fraction(padding, d)
    (sz[d] == 1 || iszero(p)) ? sz[d] : sz[d] + next_fast_fft_size(max(4, 2 * round(Int, sz[d] * abs(p))))
end

"size of the returned result: measured size where the padding is ≥ 0, padded size where it is < 0"
_result_size(sz::Tuple, sz_padded::Tuple, padding) =
    ntuple(d -> _padding_fraction(padding, d) < 0 ? sz_padded[d] : sz[d], length(sz))

"crop `x` (on the padded grid) to the result size, if that differs"
_crop_result(x, result_size) = size(x) == result_size ? x : center_extract(x, result_size)
