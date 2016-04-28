#ifndef RGM_INFERENCE_HPP_
#define RGM_INFERENCE_HPP_

#include "AOGrammar.hpp"
#include "ParseTree.hpp"

namespace RGM
{
/// Parsing with AOGrammar
class Inference
{
public:
    // control detection/parsing
    struct Param {
        Param();

        Scalar thresh_;
        bool   useNMS_;
        Scalar nmsOverlap_;
        bool   nmsDividedByUnion_;        
    };

    typedef std::map<boost::uuids::uuid, std::vector<Matrix > >    Maps;
    typedef std::map<boost::uuids::uuid, std::vector<bool > >      Status;
    typedef std::map<boost::uuids::uuid, std::vector<MatrixXi> >   argMaps;

    /// Type of a detection
    typedef Detection_<Scalar> Detection;

    /// constructor
    explicit Inference(AOGrammar & g, Param & p);

    /// Computes the detection results
    void runDetection(const Scalar thresh, cv::Mat img, Scalar & maxDetNum,
                      std::vector<ParseTree> & pt);

    void runDetection(const Scalar thresh, const FeaturePyramid & pyramid,
                      Scalar & maxDetNum,
                      std::vector<ParseTree> & pt);

    /// Computes the score maps using DP algorithm
    bool runDP(const FeaturePyramid & pyramid);

    /// Runs parsing
    void runParsing(const Scalar thresh, const FeaturePyramid & pyramid,
                    Scalar & maxDetNum, std::vector<ParseTree> & pt);

    /// Computes a parse tree
    void parse(const FeaturePyramid & pyramid, Detection & cand,
               ParseTree & pt);

private:
    /// Computes filter responses of T-nodes
    bool computeTnodeFilterResponses(const FeaturePyramid & pyramid);

    /// Computes the scale prior feature
    void computeScalePriorFeature(int nbLevels);

    /// Applies the compositional rule or the deformation rule for an AND-node
    bool computeANDNode(Node * node, int padx, int pady);

    /// Bounded DT
    void DT2D(Matrix & scoreMap, Deformation::Param & w, int shift, MatrixXi & Ix, MatrixXi & Iy);
    void DT1D(const Scalar *vals, Scalar *out_vals, int *I, int step, int shift, int n, Scalar a, Scalar b);

    /// Applies the switching rule for an OR-node
    bool computeORNode(Node * node);

    /// Parses an OR-node
    bool parseORNode(int head, std::vector<Node *> & gBFS, std::vector<int> & ptBFS,
                     const FeaturePyramid & pyramid, ParseTree & pt);

    /// Parses an AND-node
    bool parseANDNode(int head, std::vector<Node *> & gBFS, std::vector<int> & ptBFS,
                      const FeaturePyramid & pyramid, ParseTree & pt);

    /// Parses an T-node
    bool parseTNode(int idx, std::vector<Node *> & gBFS, std::vector<int> & ptBFS,
                    ParseTree & pt);

    /// Returns score maps of a node
    const std::vector<Matrix >& scoreMaps(const Node * n) ;
    std::vector<Matrix >& getScoreMaps(const Node * n);

    /// Set score maps to a node
    void setScoreMaps(const Node * n, int nbLevels, std::vector<Matrix> & s, const std::vector<bool> & validLevels);

    /// Returns the deformation maps
    std::vector<MatrixXi>& getDeformationX(const Node * n);
    std::vector<MatrixXi>& getDeformationY(const Node * n);

    /// Release memory
    void release();

private:
    AOGrammar* grammar_;
    Param*  param_;

    Matrix  scalepriorFeatures_; // a 3 * nbLevels matrix

    Maps    scoreMaps_; // for nodes in an AOG, for each level in the feature pyramid
    argMaps deformationX_;
    argMaps deformationY_;

    DEFINE_RGM_LOGGER;

}; // class Inference


} // namespace RGM

#endif // RGM_INFERENCE_HPP_
