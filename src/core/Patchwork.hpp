// This file is adapted from FFLDv2 (the Fast Fourier Linear Detector version 2)
// Copyright (c) 2013 Idiap Research Institute, <http://www.idiap.ch/>
// Written by Charles Dubout <charles.dubout@idiap.ch>

#ifndef RGM_PATCHWORK_HPP_
#define RGM_PATCHWORK_HPP_

#include <utility>

extern "C" {
#include <fftw3.h>
}

#include "FeaturePyramid.hpp"
#include "Rectangle.hpp"


namespace RGM
{
/// The Patchwork class computes full convolutions much faster using FFT
class Patchwork
{
public:
    /// Type of a patchwork plane cell (fixed-size complex vector of size NbFeatures).
    typedef Eigen::Array<CScalar, FeaturePyramid::NbFeatures, 1> Cell;

    /// Type of a patchwork plane (matrix of cells).
    typedef Eigen::Matrix<Cell, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Plane;

    /// Type of a patchwork filter (plane + original filter size).
    typedef std::pair<Plane, std::pair<int, int> > Filter;

    /// Constructs an empty patchwork. An empty patchwork has no plane.
    Patchwork();

    /// Constructs a patchwork from a pyramid.
    /// @param[in] pyramid The pyramid of features.    
    /// @note If the pyramid is larger than the last maxRows and maxCols passed to the Init method
    /// the Patchwork will be empty.
    /// @note Assumes that the features of the pyramid levels are zero in the padded regions but for
    /// the last feature, which is assumed to be one.
    Patchwork(const FeaturePyramid & pyramid);

    /// Returns the amount of horizontal zero padding (in cells).
    int padx() const;

    /// Returns the amount of vertical zero padding (in cells).
    int pady() const;

    /// Returns the number of levels per octave in the pyramid.
    int interval() const;

    /// Returns whether the patchwork is empty. An empty patchwork has no plane.
    bool empty() const;

    /// Returns the convolutions of the patchwork with filters (useful to compute the SVM margins).
    /// @param[in] filters The filters.
    /// @param[out] convolutions The convolutions (filters x levels).
    void convolve(const std::vector<Filter *> & filters,
                  std::vector<std::vector<Matrix> > & convolutions) const;

    /// Initializes the FFTW library.
    /// @param[in] maxRows Maximum number of rows of a pyramid level (including padding).
    /// @param[in] maxCols Maximum number of columns of a pyramid level (including padding).    
    /// @returns Whether the initialization was successful.
    /// @note Must be called before any other method (including constructors).
    static bool InitFFTW(int maxRows, int maxCols);

    /// Returns the current maximum number of rows of a pyramid level (including padding).
    static int MaxRows();

    /// Returns the current maximum number of columns of a pyramid level (including padding).
    static int MaxCols();

    /// Returns a transformed version of a filter to be used by the @c convolve method.
    /// @param[in] filter Filter to transform.
    /// @param[out] result Transformed filter.
    /// @note If Init was not already called or if the filter is larger than the last maxRows and
    /// maxCols passed to the Init method the result will be empty.
    static void TransformFilter(const FeaturePyramid::Level & filter, Filter & result);

private:
    int padx_;
    int pady_;
    int interval_;
    std::vector<std::pair<Rectangle2i, int> > rectangles_;
    std::vector<Plane> planes_;

    static int MaxRows_;
    static int MaxCols_;
    static int HalfCols_;

#ifndef RGM_USE_DOUBLE
    static fftwf_plan Forwards_;
    static fftwf_plan Inverse_;
#else
    static fftw_plan Forwards_;
    static fftw_plan Inverse_;
#endif

}; // class Patchwork

} // namespace RGM


// Some compilers complain about the lack of a NumTraits for Eigen::Array<CScalar, NbFeatures, 1>
namespace Eigen
{
template <>
struct NumTraits<Array<RGM::CScalar, RGM::FeaturePyramid::NbFeatures, 1> > :
        GenericNumTraits<Array<RGM::CScalar, RGM::FeaturePyramid::NbFeatures, 1> > {
    static inline RGM::Scalar dummy_precision()
    {
        return 0; // Never actually called
    }
};
} // namespace Eigen


#endif // RGM_PATCHWORK_HPP_
