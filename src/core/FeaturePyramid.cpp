#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>

#include "FeaturePyramid.hpp"
#include "UtilOpencv.hpp"
#include "UtilSerialization.hpp"
#include "UtilMath.hpp"

namespace RGM
{

FeaturePyramid::FeaturePyramid() :
    cellSize_(0), padx_(0), pady_(0), interval_(0), octave_(0), extraOctave_(false),
    imgWd_(0), imgHt_(0)
{
    mu_.setZero();
    mu_(NbFeatures-1) = 1;
}

FeaturePyramid::FeaturePyramid(const cv::Mat & image, int cellSize, int padx, int pady, int octave, int interval, bool extraOctave) :
    cellSize_(0), padx_(0), pady_(0), interval_(0), octave_(0), extraOctave_(false),
    imgWd_(0), imgHt_(0)
{
    mu_.setZero();
    mu_(NbFeatures-1) = 1;

    computePyramid(image, cellSize, padx, pady, mu_, octave, interval, extraOctave);
}

FeaturePyramid::FeaturePyramid(const cv::Mat &image, int cellSize, int padx, int pady, Cell &bgmu, int octave, int interval, bool extraOctave)
{
    mu_ = bgmu;

    computePyramid(image, cellSize, padx, pady, mu_, octave, interval, extraOctave);
}

FeaturePyramid::FeaturePyramid(int cellSize, int padx, int pady, int octave, int interval, bool extraOctave, int imgWd, int imgHt,
                               std::vector<Level> & levels, std::vector<Scalar> & scales) :
    cellSize_(0), padx_(0), pady_(0), interval_(0), octave_(0), extraOctave_(false)
{
    if ( (!extraOctave && cellSize/2<2) || (extraOctave && cellSize/4<2)  ||
         (padx < 1) || (pady < 1) || (interval < 1) ||
         levels.size() != scales.size() ) {
        RGM_LOG(error, "Attempting to create an empty or invalid pyramid" );
        return;
    }

    cellSize_ = cellSize;
    padx_ = padx;
    pady_ = pady;
    octave_ = octave;
    interval_ = interval;
    extraOctave_ = extraOctave;
    imgWd_ = imgWd;
    imgHt_ = imgHt;

    levels_.swap(levels);
    scales_.swap(scales);
}

void FeaturePyramid::computePyramid(const cv::Mat &image, int cellSize, int padx, int pady,
                                    Cell &bgmu, int octave, int interval, bool extraOctave)
{
    if (image.empty() ||  (!extraOctave && cellSize/2<2) ||
            (extraOctave && cellSize/4<2)  || (padx < 0) || (pady < 0) || (interval < 1) ) {
        RGM_LOG(error, "Attempting to create an empty pyramid" );
        return;
    }

    // Min size of a level
    const int MinLevelSz = 5;

    // Compute the number of scales such that the smallest size of the last level is MinLevelSz (=5)
    int minSz    = std::min( image.cols, image.rows );
    int levelSz  = octave>0 ? std::max( MinLevelSz, static_cast<int>(minSz/(cellSize*pow(2.0F, octave))) ) : MinLevelSz;

    // using double to be consistent with matlab
    const int maxScale = 1 + floor(log(static_cast<double>(minSz) / double(levelSz*cellSize)) / log(pow(double(2.0F), double(1.0F)/interval)));

    // Cannot compute the pyramid on images too small
    if (maxScale < interval) {
        //RGM_LOG(warning, "The input image is too small to create a pyramid");
        return;
    }

    cellSize_    = cellSize;
    padx_        = padx;
    pady_        = pady;
    octave_      = octave;
    interval_    = interval;
    extraOctave_ = extraOctave;
    imgWd_       = image.cols;
    imgHt_       = image.rows;

    int extraInterval = extraOctave_ ? interval_ : 0;

    int totalLevels = maxScale + extraInterval + interval_;

    levels_.resize(totalLevels);
    scales_.resize(totalLevels);
    validLevels_.resize(totalLevels, true);

    // Convert input image to Scalar type
    cv::Mat imgf;
    image.convertTo(imgf, CV_MAKE_TYPE(cv::DataDepth<Scalar>::value, 3));

    const Scalar sc = pow(2.0F, 1.0F / static_cast<Scalar>(interval));

#pragma omp parallel for
    for (int i = 0; i < interval; ++i) {
        Scalar scale = 1.0F / pow(sc, i);
        cv::Mat scaled;
        if ( scale==1.0 ) {
            scaled = imgf;
        } else {
            cv::resize(imgf, scaled, cv::Size(imgf.cols * scale + 0.5f, imgf.rows * scale + 0.5f), 0.0, 0.0, cv::INTER_AREA);
        }

        if (extraOctave_) {
            // Optional (cellSize/4) x (cellSize/4) features
            computeHOGFeature(scaled, levels_[i], bgmu, padx_, pady_, cellSize_/4);
            scales_[i] = scale * 4;
        }

        // First octave at twice the image resolution, i.e., (cellSize/2) x (cellSize/2) features
        computeHOGFeature(scaled, levels_[i+extraInterval], bgmu, padx_, pady_, cellSize_/2);
        scales_[i+extraInterval] = scale * 2;

        // Second octave at the original resolution, i.e., cellSize x cellSize HOG features
        int ii = i + interval_ + extraInterval;
        if ( ii < totalLevels ) {
            computeHOGFeature(scaled, levels_[ii], bgmu, padx_, pady_, cellSize_);
            scales_[ii] = scale;
        }

        // Remaining octaves
        for (int j = i+interval_;  j<maxScale; j+=interval_) {
            cv::resize(scaled, scaled, cv::Size(scaled.cols * 0.5f + 0.5f, scaled.rows * 0.5f + 0.5f), 0.0, 0.0, cv::INTER_AREA);
            ii = j + interval_ + extraInterval;
            computeHOGFeature(scaled, levels_[ii], bgmu, padx_, pady_, cellSize_);
            scales_[ii] = 0.5F * scales_[j+extraInterval];
        }
    }
}


bool FeaturePyramid::empty() const
{
    return levels().empty();
}

#if RGM_USE_PCA_DIM
bool FeaturePyramid::emptyPCA() const
{
    return PCAlevels().empty();
}
#endif

int FeaturePyramid::cellSize() const
{
    return cellSize_;
}

int FeaturePyramid::padx() const
{
    return padx_;
}

int FeaturePyramid::pady() const
{
    return pady_;
}

int FeaturePyramid::octave() const
{
    return octave_;
}

int FeaturePyramid::interval() const
{
    return interval_;
}

bool FeaturePyramid::extraOctave() const
{
    return extraOctave_;
}

const std::vector<FeaturePyramid::Level> & FeaturePyramid::levels() const
{
    return levels_;
}

const std::vector<Scalar> & FeaturePyramid::scales() const
{
    return scales_;
}

const std::vector<bool> & FeaturePyramid::validLevels() const
{
    return validLevels_;
}

std::vector<bool> & FeaturePyramid::getValidLevels()
{
    return validLevels_;
}

const std::vector<int> & FeaturePyramid::idxValidLevels()
{
    idxValidLevels_.resize(nbValidLevels());

    for ( int i=0, j=0; i<nbLevels(); ++i ) {
        if (validLevels_[i]) {
            idxValidLevels_[j++] = i;
        }
    }

    return idxValidLevels_;
}

int FeaturePyramid::nbLevels() const
{
    return levels().size();
}

int FeaturePyramid::nbValidLevels() const
{
    int n=0;
    n = std::accumulate(validLevels().begin(), validLevels().end(), n);
    return n;
}

int FeaturePyramid::imgWd() const
{
    return imgWd_;
}

int FeaturePyramid::imgHt() const
{
    return imgHt_;
}

void FeaturePyramid::convolve(const Level & filter, std::vector<Matrix> & convolutions) const
{
    convolutions.resize(levels_.size());

#pragma omp parallel for
    for (int i = 0; i < levels_.size(); ++i) {
        Convolve(levels_[i], filter, convolutions[i]);
    }
}

RGM::FeaturePyramid::Level FeaturePyramid::Flip(const FeaturePyramid::Level & level)
{
    // Symmetric features
    const int symmetry[NbFeatures] = {
        9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 17, 16, 15, 14, 13, 12, 11, 10, // Contrast-sensitive
        18, 26, 25, 24, 23, 22, 21, 20, 19, // Contrast-insensitive
        29, 30, 27, 28,  //28, 27, 30, 29, // Texture
    #if (!defined RGM_USE_EXTRA_FEATURES) || (defined RGM_USE_FELZENSZWALB_HOG_FEATURES)
        31 // Truncation
    #else
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40, // Uniform LBP
        41, 42, 43, 44, 45, 46, // Color
        47 // Truncation
    #endif
    };

    // Symmetric filter
    FeaturePyramid::Level result(level.rows(), level.cols());

    for (int y = 0; y < level.rows(); ++y)
        for (int x = 0; x < level.cols(); ++x)
            for (int i = 0; i < NbFeatures; ++i) {
                result(y, x)(i) = level(y, level.cols() - 1 - x)(symmetry[i]);
            }

    return result;
}

Eigen::Map<Matrix, Eigen::Aligned> FeaturePyramid::Map(Level & level)
{
    return Eigen::Map<Matrix, Eigen::Aligned>(level.data()->data(), level.rows(),
                                              level.cols() * NbFeatures);
}

Eigen::Map<dMatrix, Eigen::Aligned> FeaturePyramid::dMap(dLevel & level)
{
    return Eigen::Map<dMatrix, Eigen::Aligned>(level.data()->data(), level.rows(),
                                               level.cols() * NbFeatures);
}

const Eigen::Map<const Matrix, Eigen::Aligned> FeaturePyramid::Map(const Level & level)
{
    return Eigen::Map<const Matrix, Eigen::Aligned>(level.data()->data(), level.rows(),
                                                    level.cols() * NbFeatures);
}

cv::Mat_<Scalar> FeaturePyramid::convertToMat(const Level & level, int startDim, int endDim)
{
    //startDim = std::max<int>(0, startDim);
    //endDim   = std::min<int>(NbFeatures, endDim);

    DEFINE_RGM_LOGGER;

    if (endDim - startDim <= 0 ) {
        RGM_LOG(error, " Attempting to access invalid dimensions" );
        return cv::Mat_<Scalar>();
    }

    int dim[] = { level.rows(), level.cols(), endDim-startDim };
    cv::Mat_<Scalar> m(3, dim);

    for ( int d=startDim, dst=0; d<endDim; ++d, ++dst ) {
        for( int r=0; r<level.rows(); ++r ) {
            for ( int c=0; c<level.cols(); ++c ) {
                m(r, c, dst) = level(r,c)(d);
            }
        }
    }

    return m;
}

cv::Mat_<Scalar> FeaturePyramid::fold(const Level & level)
{
    return convertToMat(level, 18, 27);
}

void FeaturePyramid::resize(const Level & in, Scalar factor,  Level & out)
{
    if ( factor == 1.0F ) {
        out = in;
        return;
    }

    out = Level::Constant(ROUND(factor * in.rows()), ROUND(factor * in.cols()),
                          Cell::Zero());

    int dim[] = {in.rows(), in.cols(), FeaturePyramid::NbFeatures};

    cv::Mat_<Scalar> inMat(3, dim);
    for ( int r = 0; r < in.rows(); ++r )
        for ( int c = 0; c < in.cols(); ++c )
            for ( int d = 0; d < FeaturePyramid::NbFeatures; ++d ) {
                inMat(r, c, d) = in(r, c)(d);
            }

    cv::Mat_<Scalar> xMat = OpencvUtil_<Scalar>::resize(inMat, factor, cv::INTER_CUBIC);

    for ( int r = 0; r < out.rows(); ++r )
        for ( int c = 0; c < out.cols(); ++c )
            for ( int d = 0; d < FeaturePyramid::NbFeatures; ++d ) {
                out(r, c)(d) = xMat(r, c, d);
            }

}

cv::Mat FeaturePyramid::visualize(const Level & level, int bs)
{
    // Make pictures of positive and negative weights
    cv::Mat_<Scalar> w = convertToMat(level, 0, 9);

    Scalar maxVal = *(std::max_element(w.begin(), w.end()));
    Scalar minVal = *(std::min_element(w.begin(), w.end()));

    /*int dim[] = { w.size[0], w.size[1]*w.size[2]};
    cv::Mat w1 = cv::Mat(w).reshape(1, 2, dim); // opencv not implement this for different dims

    double minVal, maxVal;
    cv::minMaxLoc(w1, &minVal, &maxVal);*/
    Scalar scale = std::max<double>(maxVal, -minVal);

    cv::Mat pos = OpencvUtil::pictureHOG(w, bs) * 255.0F / scale;

    cv::Mat neg;
    if ( minVal<0 ) {
        cv::Mat_<Scalar> minusw = w * -1.0F;
        neg = OpencvUtil::pictureHOG(minusw, bs) * 255.0F / scale;
    }

    // Put pictures together and draw
    int buff = 10;
    cv::Mat img(pos.rows+2*buff, pos.cols+neg.cols+4*buff, pos.type(), cv::Scalar::all(128));

    pos.copyTo( img(cv::Rect(buff, buff, pos.cols, pos.rows))  );

    if ( minVal<0 ) {
        neg.copyTo( img(cv::Rect(pos.cols+2*buff, buff, neg.cols, neg.rows)) );
    }

    cv::Mat imgShow;
    cv::normalize(img, imgShow, 255, 0.0, cv::NORM_MINMAX, CV_8UC1);
    cv::String winName("HOG");
    cv::imshow(winName, imgShow);
    cv::waitKey(0);

    return imgShow;
}

int FeaturePyramid::VirtualPadding(int padding, int ds)
{
    // subtract one because each level already has a one padding wide border around it
    return padding*(std::pow(2, ds)-1);
}

namespace detail
{
struct HOGTable {
    char bins[512][512][2];
    Scalar magnitudes[512][512][2];

    // Singleton pattern
    static const HOGTable & Singleton()
    {
        return Singleton_;
    }

private:
    // Singleton pattern
    HOGTable() throw ()
    {
        for (int dy = -255; dy <= 255; ++dy) {
            for (int dx = -255; dx <= 255; ++dx) {
                // Magnitude in the range [0, 1]
                const double magnitude = sqrt(dx * dx + dy * dy) / 255.0;

                // Angle in the range [-pi, pi]
                double angle = atan2(static_cast<double>(dy), static_cast<double>(dx));

                // Convert it to the range [9.0, 27.0]
                angle = angle * (9.0 / M_PI) + 18.0;

                // Convert it to the range [0, 18)
                if (angle >= 18.0) {
                    angle -= 18.0;
                }

                // Bilinear interpolation
                const int bin0 = angle;
                const int bin1 = (bin0 < 17) ? (bin0 + 1) : 0;
                const double alpha = angle - bin0;

                bins[dy + 255][dx + 255][0] = bin0;
                bins[dy + 255][dx + 255][1] = bin1;
                magnitudes[dy + 255][dx + 255][0] = magnitude * (1.0 - alpha);
                magnitudes[dy + 255][dx + 255][1] = magnitude * alpha;
            }
        }
    }

    // Singleton pattern
    HOGTable(const HOGTable &) throw ();
    void operator=(const HOGTable &) throw ();

    static const HOGTable Singleton_;
}; // struct HOGTable

const HOGTable HOGTable::Singleton_;

} // namespace detail

#ifndef RGM_USE_FELZENSZWALB_HOG_FEATURES
void FeaturePyramid::computeHOGFeature(const cv::Mat & image, Level & level, Cell & bgmu, int padx, int pady, int cellSize)
{
    DEFINE_RGM_LOGGER;

    // Get the size of image
    const int width  = image.cols;
    const int height = image.rows;
    const int depth  = image.channels();

    // Make sure the image is big enough
    if ((width < cellSize) || (height < cellSize) || (depth != 3 ) || (padx < 0) || (pady < 0) || (cellSize < 2)) {
        level.swap(Level());
        RGM_LOG(error, "Attempting to compute an empty pyramid level" );
        return;
    }

    bool zeroPadx = (padx==0);
    bool zeroPady = (pady==0);

    if (zeroPadx) {
        padx = 1;
    }

    if ( zeroPady ) {
        pady = 1;
    }

    // Resize the feature matrix
    level = Level::Constant((height + cellSize / 2) / cellSize + 2 * pady,
                            (width + cellSize / 2) / cellSize + 2 * padx, Cell::Zero());

    const Scalar invCellSize = static_cast<Scalar>(1) / cellSize;

    for (int y = 0; y < height; ++y) {
        const int yabove = std::max(y - 1, 0);
        const int ybelow = std::min(y + 1, height - 1);

        for (int x = 0; x < width; ++x) {
            const int xright = std::min(x + 1, width - 1);
            const int xleft = std::max(x - 1, 0);

            const Pixel & pixelyp = image.at<Pixel>(ybelow, x);
            const Pixel & pixelym = image.at<Pixel>(yabove, x);
            const Pixel & pixelxp = image.at<Pixel>(y, xright);
            const Pixel & pixelxm = image.at<Pixel>(y, xleft);

            // Use the channel with the largest gradient magnitude
            int maxMagnitude = 0;
            int argDx = 255;
            int argDy = 255;

            for (int i = 0; i < depth; ++i) {
                const int dx = static_cast<int>( pixelxp[i] - pixelxm[i] );
                const int dy = static_cast<int>( pixelyp[i] - pixelym[i] );

                if (dx * dx + dy * dy > maxMagnitude) {
                    maxMagnitude = dx * dx + dy * dy;
                    argDx = dx + 255;
                    argDy = dy + 255;
                }
            }

            const char bin0 = detail::HOGTable::Singleton().bins[argDy][argDx][0];
            const char bin1 = detail::HOGTable::Singleton().bins[argDy][argDx][1];
            const Scalar magnitude0 = detail::HOGTable::Singleton().magnitudes[argDy][argDx][0];
            const Scalar magnitude1 = detail::HOGTable::Singleton().magnitudes[argDy][argDx][1];

            // Bilinear interpolation
            const Scalar xp = (x + static_cast<Scalar>(0.5)) * invCellSize + padx - 0.5f;
            const Scalar yp = (y + static_cast<Scalar>(0.5)) * invCellSize + pady - 0.5f;
            const int ixp = xp;
            const int iyp = yp;
            const Scalar xp0 = xp - ixp;
            const Scalar yp0 = yp - iyp;
            const Scalar xp1 = 1 - xp0;
            const Scalar yp1 = 1 - yp0;

            level(iyp    , ixp    )(bin0) += xp1 * yp1 * magnitude0;
            level(iyp    , ixp    )(bin1) += xp1 * yp1 * magnitude1;
            level(iyp    , ixp + 1)(bin0) += xp0 * yp1 * magnitude0;
            level(iyp    , ixp + 1)(bin1) += xp0 * yp1 * magnitude1;
            level(iyp + 1, ixp    )(bin0) += xp1 * yp0 * magnitude0;
            level(iyp + 1, ixp    )(bin1) += xp1 * yp0 * magnitude1;
            level(iyp + 1, ixp + 1)(bin0) += xp0 * yp0 * magnitude0;
            level(iyp + 1, ixp + 1)(bin1) += xp0 * yp0 * magnitude1;

#ifdef RGM_USE_EXTRA_FEATURES
            // Normalize by the number of pixels
            const Scalar normalization = 2.0 / (cellSize * cellSize);

            // Texture (Uniform LBP) features
            const int LBP_TABLE[256] = {
                0, 1, 1, 2, 1, 9, 2, 3, 1, 9, 9, 9, 2, 9, 3, 4, 1, 9, 9, 9, 9, 9, 9, 9,
                2, 9, 9, 9, 3, 9, 4, 5, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                2, 9, 9, 9, 9, 9, 9, 9, 3, 9, 9, 9, 4, 9, 5, 6, 1, 9, 9, 9, 9, 9, 9, 9,
                9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                2, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 3, 9, 9, 9, 9, 9, 9, 9,
                4, 9, 9, 9, 5, 9, 6, 7, 1, 2, 9, 3, 9, 9, 9, 4, 9, 9, 9, 9, 9, 9, 9, 5,
                9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 6, 9, 9, 9, 9, 9, 9, 9, 9,
                9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7,
                2, 3, 9, 4, 9, 9, 9, 5, 9, 9, 9, 9, 9, 9, 9, 6, 9, 9, 9, 9, 9, 9, 9, 9,
                9, 9, 9, 9, 9, 9, 9, 7, 3, 4, 9, 5, 9, 9, 9, 6, 9, 9, 9, 9, 9, 9, 9, 7,
                4, 5, 9, 6, 9, 9, 9, 7, 5, 6, 9, 7, 6, 7, 7, 8
            };

            // Use the green channel
            const Pixel & pixel = image.at<Pixel>(y, x);
            const Scalar      g = pixel[1];

            // clock-wise pixels in 8 negihborhood of (x,y)
            const int lbp = (static_cast<int>(image.at<Pixel>(yabove, xleft)[1] >= g)) |
                    (static_cast<int>(image.at<Pixel>(yabove, x     )[1] >= g) << 1) |
                    (static_cast<int>(image.at<Pixel>(yabove, xright)[1] >= g) << 2) |
                    (static_cast<int>(image.at<Pixel>(y,      xright)[1] >= g) << 3) |
                    (static_cast<int>(image.at<Pixel>(ybelow, xright)[1] >= g) << 4) |
                    (static_cast<int>(image.at<Pixel>(ybelow, x     )[1] >= g) << 5) |
                    (static_cast<int>(image.at<Pixel>(ybelow, xleft )[1] >= g) << 6) |
                    (static_cast<int>(image.at<Pixel>(y,      xleft )[1] >= g) << 7);

            // Bilinear interpolation
            level(iyp    , ixp    )(LBP_TABLE[lbp] + 31) += xp1 * yp1 * normalization;
            level(iyp    , ixp + 1)(LBP_TABLE[lbp] + 31) += xp0 * yp1 * normalization;
            level(iyp + 1, ixp    )(LBP_TABLE[lbp] + 31) += xp1 * yp0 * normalization;
            level(iyp + 1, ixp + 1)(LBP_TABLE[lbp] + 31) += xp0 * yp0 * normalization;

            // Color features
            if (depth >= 3) {
                const Scalar r = pixel[2] * static_cast<Scalar>(1.0 / 255.0);
                const Scalar g = pixel[1] * static_cast<Scalar>(1.0 / 255.0);
                const Scalar b = pixel[0] * static_cast<Scalar>(1.0 / 255.0);

                const Scalar minRGB = std::min(r, std::min(g, b));
                const Scalar maxRGB = std::max(r, std::max(g, b));
                const Scalar chroma = maxRGB - minRGB;

                if (chroma > 0.05) {
                    Scalar hue = 0;

                    if (r == maxRGB) {
                        hue = (g - b) / chroma;
                    } else if (g == maxRGB) {
                        hue = (b - r) / chroma + 2;
                    } else {zeroPadx
                        hue = (r - g) / chroma + 4;
                    }

                    if (hue < 0) {
                        hue += 6;
                    } else if (hue >= 6) {
                        hue = 0;
                    }

                    const Scalar saturation = chroma / maxRGB;

                    // Bilinear interpolation
                    const int bin0 = hue;
                    const int bin1 = (hue < 5) ? (hue + 1) : 0; // (hue0 < 5) ? (hue0 + 1) : 0;
                    const Scalar alpha = hue - bin0;
                    const Scalar magnitude0 = saturation * normalization * (1 - alpha);
                    const Scalar magnitude1 = saturation * normalization * alpha;

                    level(iyp    , ixp    )(bin0 + 41) += xp1 * yp1 * magnitude0;
                    level(iyp    , ixp    )(bin1 + 41) += xp1 * yp1 * magnitude1;
                    level(iyp    , ixp + 1)(bin0 + 41) += xp0 * yp1 * magnitude0;
                    level(iyp    , ixp + 1)(bin1 + 41) += xp0 * yp1 * magnitude1;
                    level(iyp + 1, ixp    )(bin0 + 41) += xp1 * yp0 * magnitude0;
                    level(iyp + 1, ixp    )(bin1 + 41) += xp1 * yp0 * magnitude1;
                    level(iyp + 1, ixp + 1)(bin0 + 41) += xp0 * yp0 * magnitude0;
                    level(iyp + 1, ixp + 1)(bin1 + 41) += xp0 * yp0 * magnitude1;
                }
            }
#endif
        }
    }

    // Compute the "gradient energy" of each cell, i.e. ||C(i,j)||^2
    for (int y = 0; y < level.rows(); ++y) {
        for (int x = 0; x < level.cols(); ++x) {
            Scalar sumSq = 0;

            for (int i = 0; i < 9; ++i)
                sumSq += (level(y, x)(i) + level(y, x)(i + 9)) *
                         (level(y, x)(i) + level(y, x)(i + 9));

            level(y, x)(NbFeatures - 1) = sumSq;
        }
    }

    // Compute the four normalization factors then normalize and clamp everything
    const Scalar EPS = std::numeric_limits<Scalar>::epsilon();

    for (int y = pady; y < level.rows() - pady; ++y) {
        for (int x = padx; x < level.cols() - padx; ++x) {
            const Scalar n0 = 1 / sqrt(level(y - 1, x - 1)(NbFeatures - 1) +
                                       level(y - 1, x    )(NbFeatures - 1) +
                                       level(y    , x - 1)(NbFeatures - 1) +
                                       level(y    , x    )(NbFeatures - 1) + EPS);
            const Scalar n1 = 1 / sqrt(level(y - 1, x    )(NbFeatures - 1) +
                                       level(y - 1, x + 1)(NbFeatures - 1) +
                                       level(y    , x    )(NbFeatures - 1) +
                                       level(y    , x + 1)(NbFeatures - 1) + EPS);
            const Scalar n2 = 1 / sqrt(level(y    , x - 1)(NbFeatures - 1) +
                                       level(y    , x    )(NbFeatures - 1) +
                                       level(y + 1, x - 1)(NbFeatures - 1) +
                                       level(y + 1, x    )(NbFeatures - 1) + EPS);
            const Scalar n3 = 1 / sqrt(level(y    , x    )(NbFeatures - 1) +
                                       level(y    , x + 1)(NbFeatures - 1) +
                                       level(y + 1, x    )(NbFeatures - 1) +
                                       level(y + 1, x + 1)(NbFeatures - 1) + EPS);

            // Contrast-insensitive features
            for (int i = 0; i < 9; ++i) {
                const Scalar sum = level(y, x)(i) + level(y, x)(i + 9);
                const Scalar h0 = std::min(sum * n0, static_cast<Scalar>(0.2));
                const Scalar h1 = std::min(sum * n1, static_cast<Scalar>(0.2));
                const Scalar h2 = std::min(sum * n2, static_cast<Scalar>(0.2));
                const Scalar h3 = std::min(sum * n3, static_cast<Scalar>(0.2));
                level(y, x)(i + 18) = (h0 + h1 + h2 + h3) * static_cast<Scalar>(0.5);
            }

            // Contrast-sensitive features
            Scalar t0 = 0;
            Scalar t1 = 0;
            Scalar t2 = 0;
            Scalar t3 = 0;

            for (int i = 0; i < 18; ++i) {
                const Scalar sum = level(y, x)(i);
                const Scalar h0 = std::min(sum * n0, static_cast<Scalar>(0.2));
                const Scalar h1 = std::min(sum * n1, static_cast<Scalar>(0.2));
                const Scalar h2 = std::min(sum * n2, static_cast<Scalar>(0.2));
                const Scalar h3 = std::min(sum * n3, static_cast<Scalar>(0.2));
                level(y, x)(i) = (h0 + h1 + h2 + h3) * static_cast<Scalar>(0.5);
                t0 += h0;
                t1 += h1;
                t2 += h2;
                t3 += h3;
            }

            // Texture features
            level(y, x)(27) = t0 * static_cast<Scalar>(0.2357);
            level(y, x)(28) = t1 * static_cast<Scalar>(0.2357);
            level(y, x)(29) = t2 * static_cast<Scalar>(0.2357);
            level(y, x)(30) = t3 * static_cast<Scalar>(0.2357);
        }
    }

    // Truncation features
    if ( !zeroPadx || !zeroPady ) {
        for (int y = 0; y < level.rows(); ++y) {
            for (int x = 0; x < level.cols(); ++x) {
                if ((y < pady) || (y >= level.rows() - pady) ||
                        (x < padx) || (x >= level.cols() - padx)) {
                    level(y, x) = bgmu;
                } else {
                    level(y, x)(NbFeatures - 1) = 0;
                }
            }
        }
    }

    if ( zeroPadx || zeroPady ) {
        int x = zeroPadx ? padx + 1 : 0;
        int y = zeroPady ? pady + 1 : 0;

        Level tmp = level.block(y, x, level.rows()-2*y, level.cols()-2*x);
        level.swap(tmp);
    }

    if ( zeroPadx && zeroPady ) {
        for (int y = 0; y < level.rows(); ++y) {
            for (int x = 0; x < level.cols(); ++x) {
                level(y, x)(NbFeatures - 1) = 0;
            }
        }
    }
}
#else
void FeaturePyramid::computeHOGFeature(const cv::Mat & image, Level & level, Cell & bgmu, int padx, int pady, int cellSize)
{
    DEFINE_RGM_LOGGER;

    // Adapted from voc-release4.01/features.cc
    const Scalar EPS = 0.0001;

    const Scalar UU[9] = {
        1.0000, 0.9397, 0.7660, 0.5000, 0.1736,-0.1736,-0.5000,-0.7660,-0.9397
    };

    const Scalar VV[9] = {
        0.0000, 0.3420, 0.6428, 0.8660, 0.9848, 0.9848, 0.8660, 0.6428, 0.3420
    };

    // Get all the image members
    const int width = image.cols;
    const int height = image.rows;

    // Make sure the image is big enough
    RGM_CHECK_GE(width,  cellSize / 2);
    RGM_CHECK_GE(height, cellSize / 2);
    RGM_CHECK_GE(padx, 0);
    RGM_CHECK_GE(pady, 0);
    RGM_CHECK(((cellSize == 8) || (cellSize == 4) || (cellSize==2)), error);

    // Memory for caching orientation histograms & their norms
    int blocks[2];
    blocks[0] = floor(static_cast<double>(height) / cellSize + 0.5);
    blocks[1] = floor(static_cast<double>(width) / cellSize + 0.5);
    Eigen::MatrixXf hist = Eigen::MatrixXf::Zero(blocks[0], blocks[1] * 18);
    Eigen::MatrixXf norm = Eigen::MatrixXf::Zero(blocks[0], blocks[1]);

    // Memory for HOG features
    int out[3];
    out[0] = std::max(blocks[0] - 2, 0);
    out[1] = std::max(blocks[1] - 2, 0);
    out[2] = 27 + 4 + 1;
    RGM_CHECK_EQ(out[2], NbFeatures);

    level = Level::Constant(out[0], out[1], Cell::Zero());

    int visible[2];
    visible[0] = blocks[0] * cellSize;
    visible[1] = blocks[1] * cellSize;

    for (int x = 1; x < visible[1] - 1; ++x) {
        for (int y = 1; y < visible[0] - 1; ++y) {
            const int x2 = std::min(x, width - 2);
            const int y2 = std::min(y, height - 2);

            // Use the channel with the largest gradient magnitude
            Scalar magnitude = 0;
            Scalar argDx = 0;
            Scalar argDy = 0;

            const Pixel & pixelyp = image.at<Pixel>(y2+1, x2);
            const Pixel & pixelym = image.at<Pixel>(y2-1, x2);
            const Pixel & pixelxp = image.at<Pixel>(y2, x2+1);
            const Pixel & pixelxm = image.at<Pixel>(y2, x2-1);

            for (int i = 2; i >= 0; i--) {
                const Scalar dx = pixelxp[i] - pixelxm[i];
                const Scalar dy = pixelyp[i] - pixelym[i];

                Scalar tmp = dx * dx + dy * dy;

                if ( tmp > magnitude) {
                    magnitude = tmp;
                    argDx = dx;
                    argDy = dy;
                }
            }

            // Snap to one of 18 orientations
            int theta = 0;

            Scalar best = 0;

            for (int i = 0; i < 9; ++i) {
                Scalar dot = UU[i] * argDx + VV[i] * argDy;

                if (dot > best) {
                    best = dot;
                    theta = i;
                } else if (-dot > best) {
                    best = -dot;
                    theta = i + 9;
                }
            }

            // Add to 4 histograms around pixel using linear interpolation
            Scalar xp = (x + Scalar(0.5)) / (Scalar)cellSize - Scalar(0.5);
            Scalar yp = (y + Scalar(0.5)) / (Scalar)cellSize - Scalar(0.5);
            int ixp = floor(xp);
            int iyp = floor(yp);
            Scalar vx0 = xp - ixp;
            Scalar vy0 = yp - iyp;
            Scalar vx1 = 1 - vx0;
            Scalar vy1 = 1 - vy0;

            magnitude = sqrt(magnitude);

            if ((ixp >= 0) && (iyp >= 0)) {
                hist(iyp, ixp * 18 + theta) += vx1 * vy1 * magnitude;
            }

            if ((ixp + 1 < blocks[1]) && (iyp >= 0)) {
                hist(iyp, (ixp + 1) * 18 + theta) += vx0 * vy1 * magnitude;
            }

            if ((ixp >= 0) && (iyp + 1 < blocks[0])) {
                hist(iyp + 1, ixp * 18 + theta) += vx1 * vy0 * magnitude;
            }

            if ((ixp + 1 < blocks[1]) && (iyp + 1 < blocks[0])) {
                hist(iyp + 1, (ixp + 1) * 18 + theta) += vx0 * vy0 * magnitude;
            }
        }
    }

    // Compute energy in each block by summing over orientations
    for (int y = 0; y < blocks[0]; ++y) {
        for (int x = 0; x < blocks[1]; ++x) {
            Scalar sumSq = 0;

            for (int i = 0; i < 9; ++i)
                sumSq += ((hist(y, x * 18 + i) + hist(y, x * 18 + i + 9)) *
                          (hist(y, x * 18 + i) + hist(y, x * 18 + i + 9)));

            norm(y, x) = sumSq;
        }
    }

    for (int y = 0; y < out[0]; ++y) {
        for (int x = 0; x < out[1]; ++x) {
            // Normalization factors
            const Scalar n0 = 1 / sqrt(norm(y + 1, x + 1) + norm(y + 1, x + 2) +
                                       norm(y + 2, x + 1) + norm(y + 2, x + 2) + EPS);
            const Scalar n1 = 1 / sqrt(norm(y    , x + 1) + norm(y    , x + 2) +
                                       norm(y + 1, x + 1) + norm(y + 1, x + 2) + EPS);
            const Scalar n2 = 1 / sqrt(norm(y + 1, x    ) + norm(y + 1, x + 1) +
                                       norm(y + 2, x    ) + norm(y + 2, x + 1) + EPS);
            const Scalar n3 = 1 / sqrt(norm(y    , x    ) + norm(y    , x + 1) +
                                       norm(y + 1, x    ) + norm(y + 1, x + 1) + EPS);

            // Contrast-sensitive features
            Scalar t0 = 0;
            Scalar t1 = 0;
            Scalar t2 = 0;
            Scalar t3 = 0;

            for (int i = 0; i < 18; ++i) {
                const Scalar sum = hist(y + 1, (x + 1) * 18 + i);
                const Scalar h0 = std::min(sum * n0, Scalar(0.2));
                const Scalar h1 = std::min(sum * n1, Scalar(0.2));
                const Scalar h2 = std::min(sum * n2, Scalar(0.2));
                const Scalar h3 = std::min(sum * n3, Scalar(0.2));
                level(y, x)(i) = (h0 + h1 + h2 + h3) / 2.0;
                t0 += h0;
                t1 += h1;
                t2 += h2;
                t3 += h3;
            }

            // Contrast-insensitive features
            for (int i = 0; i < 9; ++i) {
                const Scalar sum = hist(y + 1, (x + 1) * 18 + i) +
                        hist(y + 1, (x + 1) * 18 + i + 9);
                const Scalar h0 = std::min(sum * n0, Scalar(0.2));
                const Scalar h1 = std::min(sum * n1, Scalar(0.2));
                const Scalar h2 = std::min(sum * n2, Scalar(0.2));
                const Scalar h3 = std::min(sum * n3, Scalar(0.2));
                level(y, x)(i + 18) = (h0 + h1 + h2 + h3) / 2.0F;
            }

            // Texture features
            level(y, x)(27) = t0 * Scalar(0.2357);
            level(y, x)(28) = t1 * Scalar(0.2357);
            level(y, x)(29) = t2 * Scalar(0.2357);
            level(y, x)(30) = t3 * Scalar(0.2357);

            // Truncation feature
            level(y, x)(31) = 0;
        }
    }

    // Add padding
    if ( padx>0 && pady>0 ) {
        // add 1 to padding because feature generation deletes a 1-cell wide border around the feature map
        Level tmp = Level::Constant(level.rows() + (pady + 1) * 2,
                                    level.cols() + (padx + 1) * 2, bgmu);

        tmp.block(pady + 1, padx + 1, level.rows(), level.cols()) = level;

        level.swap(tmp);
    }
}
#endif

void FeaturePyramid::Convolve(const Level & x, const Level & y, Matrix & z)
{
    // Nothing to do if x is smaller than y
    if ((x.rows() < y.rows()) || (x.cols() < y.cols())) {
        z = Matrix();
        return;
    }

    z = Matrix::Zero(x.rows() - y.rows() + 1, x.cols() - y.cols() + 1);

    for (int i = 0; i < z.rows(); ++i) {
        for (int j = 0; j < y.rows(); ++j) {
            const Eigen::Map<const Matrix, Eigen::Aligned, Eigen::OuterStride<NbFeatures> >
                    mapx(reinterpret_cast<const Scalar *>(x.row(i + j).data()), z.cols(),
                         y.cols() * NbFeatures);
#ifndef RGM_USE_DOUBLE
            const Eigen::Map<const Eigen::RowVectorXf, Eigen::Aligned>
        #else
            const Eigen::Map<const Eigen::RowVectorXd, Eigen::Aligned>
        #endif
                    mapy(reinterpret_cast<const Scalar *>(y.row(j).data()), y.cols() * NbFeatures);

            z.row(i).noalias() += mapy * mapx.transpose();
        }
    }
}

template<class Archive>
void FeaturePyramid::serialize(Archive & ar, const unsigned int version)
{
    ar & BOOST_SERIALIZATION_NVP(cellSize_);
    ar & BOOST_SERIALIZATION_NVP(padx_);
    ar & BOOST_SERIALIZATION_NVP(pady_);
    ar & BOOST_SERIALIZATION_NVP(interval_);
    ar & BOOST_SERIALIZATION_NVP(octave_);
    ar & BOOST_SERIALIZATION_NVP(extraOctave_);
    ar & BOOST_SERIALIZATION_NVP(levels_);
    ar & BOOST_SERIALIZATION_NVP(scales_);
    ar & BOOST_SERIALIZATION_NVP(imgWd_);
    ar & BOOST_SERIALIZATION_NVP(imgHt_);

}

INSTANTIATE_BOOST_SERIALIZATION(FeaturePyramid);

} // namespace RGM
