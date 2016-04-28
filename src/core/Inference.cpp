#include <map>
#include <opencv2/core/core_c.h>

#include "Inference.hpp"
#include "UtilGeneric.hpp"

namespace RGM
{

// ------- Inference::Param -------

Inference::Param::Param() :
    thresh_(0.0F), useNMS_(false), nmsOverlap_(0.5F), nmsDividedByUnion_(false)
{
}

// ------- Inference -------

Inference::Inference(AOGrammar & g, Param & p)
{
    grammar_ = &g;
    param_ = &p;
}

void Inference::runDetection(const Scalar thresh, cv::Mat img, Scalar & maxDetNum,
                             std::vector<ParseTree> & pt)
{
    if (maxDetNum <= 0) {
        return;
    }

    AOGrammar &g(*grammar_);

    FeaturePyramid pyr(img, g.cellSize(), g.padx(), g.pady(), 0, g.interval(), g.extraOctave());
    if ( pyr.empty() ) {
        return;
    }

    runDetection(thresh, pyr, maxDetNum, pt);
}

void Inference::runDetection(const Scalar thresh, const FeaturePyramid & pyramid, Scalar & maxDetNum,
                             std::vector<ParseTree> & pt)
{
    if (maxDetNum <= 0) {
        return;
    }

    if ( !runDP(pyramid) ) {
        return;
    }

    runParsing(thresh, pyramid, maxDetNum, pt);
}

bool Inference::runDP(const FeaturePyramid & pyramid)
{    
    return true;
}

void Inference::runParsing(const Scalar thresh, const FeaturePyramid & pyramid,
                           Scalar & maxDetNum, std::vector<ParseTree> & pt)
{

}

void Inference::parse(const FeaturePyramid & pyramid, Detection & cand,
                      ParseTree & pt)
{   

}

bool Inference::computeTnodeFilterResponses(const FeaturePyramid & pyramid)
{

}

void Inference::computeScalePriorFeature(int nbLevels)
{
    Scaleprior::Param tmp;
    scalepriorFeatures_ = Matrix::Zero(tmp.cols(), nbLevels);

    int s = 0;
    int e = std::min<int>(nbLevels, grammar_->interval());
    scalepriorFeatures_.block(0, s, 1, e).fill(1);

    s = e;
    e = std::min<int>(nbLevels, e*2);
    scalepriorFeatures_.block(1, s, 1, e-s).fill(1);

    s = e;
    scalepriorFeatures_.block(2, s, 1, nbLevels-s).fill(1);
}

bool Inference::computeANDNode(Node * node, int padx, int pady)
{

    return true;
}

void Inference::DT2D(Matrix & scoreMap, Deformation::Param & w,
                     int shift, MatrixXi & Ix, MatrixXi & Iy)
{

}

void Inference::DT1D(const Scalar *vals, Scalar *out_vals, int *I, int step,
                     int shift, int n, Scalar a, Scalar b)
{

}

bool Inference::computeORNode(Node * node)
{    

    return true;
}


bool Inference::parseORNode(int idx, std::vector<Node *> & gBFS, std::vector<int> & ptBFS,
                            const FeaturePyramid & pyramid, ParseTree & pt)
{

    return true;
}

bool Inference::parseANDNode(int idx, std::vector<Node *> & gBFS, std::vector<int> & ptBFS,
                             const FeaturePyramid & pyramid, ParseTree & pt)
{
    return true;
}

bool Inference::parseTNode(int idx, std::vector<Node *> & gBFS, std::vector<int> & ptBFS,
                           ParseTree & pt)
{    
    return true;
}

const std::vector<Matrix >& Inference::scoreMaps(const Node * n)
{
    RGM_CHECK_NOTNULL(n);

    if ( n->type() == Node::OR_NODE &&
         n->outEdges().size() == 1 &&
         n->outEdges()[0]->type() == Edge::SWITCHING )  {

        return scoreMaps(n->outEdges()[0]->toNode());
    }

    if (n->type() == Node::AND_NODE &&
            n->outEdges().size() == 1 &&
            n->outEdges()[0]->type() == Edge::TERMINATION ) {

        return scoreMaps(n->outEdges()[0]->toNode());
    }

    Maps::const_iterator iter = scoreMaps_.find(n->tag());
    RGM_CHECK_NOTEQ( iter,  scoreMaps_.end() );

    return iter->second;
}

std::vector<Matrix >& Inference::getScoreMaps(const Node * n)
{
    RGM_CHECK_NOTNULL(n);

    if ( n->type() == Node::OR_NODE &&
         n->outEdges().size() == 1 &&
         n->outEdges()[0]->type() == Edge::SWITCHING )  {

        return getScoreMaps(n->outEdges()[0]->toNode());
    }

    if (n->type() == Node::AND_NODE &&
            n->outEdges().size() == 1 &&
            n->outEdges()[0]->type() == Edge::TERMINATION ) {

        return getScoreMaps(n->outEdges()[0]->toNode());
    }

    Maps::iterator iter = scoreMaps_.find(n->tag());
    if ( iter == scoreMaps_.end() ) {
        scoreMaps_.insert(std::make_pair(n->tag(), std::vector<Matrix >()));
    }

    return scoreMaps_[n->tag()];
}

void Inference::setScoreMaps(const Node * n, int nbLevels, std::vector<Matrix> & s,
                             const std::vector<bool> & validLevels)
{
    RGM_CHECK_NOTNULL(n);

    if (n->type() != Node::T_NODE ) {
        return;
    }

    std::vector<Matrix >& m( getScoreMaps(n) );

    m.resize(nbLevels);

    for ( int i=0, j=0; i<nbLevels; ++i) {
        if ( validLevels[i] ) {
            m[i].swap(s[j]);
            ++j;
        } else {
            m[i] = Matrix::Constant(1, 1, -std::numeric_limits<Scalar>::infinity());
        }
    }
}

std::vector<MatrixXi>& Inference::getDeformationX(const Node * n)
{
    RGM_CHECK_NOTNULL(n);

    argMaps::iterator iter = deformationX_.find(n->tag());
    if ( iter == deformationX_.end() ) {
        deformationX_.insert(std::make_pair(n->tag(), std::vector<MatrixXi >()));
    }

    return deformationX_[n->tag()];
}

std::vector<MatrixXi>& Inference::getDeformationY(const Node * n)
{
    RGM_CHECK_NOTNULL(n);

    argMaps::iterator iter = deformationY_.find(n->tag());
    if ( iter == deformationY_.end() ) {
        deformationY_.insert(std::make_pair(n->tag(), std::vector<MatrixXi >()));
    }

    return deformationY_[n->tag()];
}

void Inference::release()
{
    scoreMaps_.clear();
    deformationX_.clear();
    deformationY_.clear();
}


} // namespace RGM
