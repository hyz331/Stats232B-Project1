#ifndef RGM_PARSETREE_HPP_
#define RGM_PARSETREE_HPP_

#include "Parameters.hpp"
#include "UtilLog.hpp"

namespace RGM
{
/// Predeclaration
class AOGrammar;
class Node;

class PtNode;
class ParseTree;


/// The PtEdge represents an instance of an Edge in a grammar
class PtEdge
{
public:
    /// Index
    enum {
        IDX_MYSELF=0, IDX_FROM, IDX_TO, IDX_G, IDX_NUM
    };

    /// Type of index
    typedef Eigen::Matrix<int, 1, IDX_NUM, Eigen::RowMajor> Index;

    /// Default constructor
    PtEdge();

    /// Destructor
    ~PtEdge();

    /// Copy constructor
    explicit PtEdge(const PtEdge & e);

    /// Constructs an Edge with given type @p fromNode and @p toNode
    explicit PtEdge(int fromNode, int toNode);

    /// Constructs an Edge with given type @p fromNode and @p toNode
    explicit PtEdge(int fromNode, int toNode, int gEdge);

    /// Returns the set of indice
    const Index & idx() const;
          Index & getIdx();

    /// Returns the fromNode
    const PtNode *fromNode(const ParseTree &pt) const;
          PtNode *getFromNode(ParseTree &pt);

    /// Returns the fromNode
    const PtNode *toNode(const ParseTree &pt) const;
          PtNode *getToNode(ParseTree &pt);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    Index idx_;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version);

}; // class PtEdge


/// The PtNode represents an instance of a Node in a grammar
class PtNode
{
public:
    /// Index
    enum {
        IDX_MYSELF=0, IDX_TYPE, IDX_BIAS, IDX_DEF, IDX_SCALEPRIOR, IDX_APP,
        IDX_G, IDX_PARSEINFO, IDX_VALID, IDX_NUM
    };

    /// Type of index
    typedef Eigen::Matrix<int, 1, IDX_NUM, Eigen::RowMajor> Index;

    /// Default constructor
    PtNode();

    /// Destructor
    ~PtNode();

    /// Copy constructor
    PtNode(const PtNode & n);

    /// Constructs a PtNode with given index @p gNode of a Node in a grammar
    PtNode(int gNode);

    /// Returns the indice of in-edges
    const std::vector<int> & idxInEdges() const;
    std::vector<int> & getIdxInEdges();

    /// Returns the indice of out-edges
    const std::vector<int> & idxOutEdges() const;
    std::vector<int> & getIdxOutEdges();

    /// Returns an InEdge with idxInEdge_[i]
    const PtEdge * inEdge(int i, const ParseTree & pt) const;
          PtEdge * getInEdge(int i, ParseTree & pt);

    /// Returns an OutEdge with idxOutEdge_[i]
    const PtEdge * outEdge(int i, const ParseTree & pt) const;
          PtEdge * getOutEdge(int i, ParseTree & pt);

    /// Returns the set of indice
    const Index & idx() const;
          Index & getIdx();

    /// Returns the parse info
    const ParseInfo *  parseInfo(const ParseTree * pt) const;
    ParseInfo *& getParseInfo(ParseTree * pt);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    std::vector<int> idxInEdges_;
    std::vector<int> idxOutEdges_;

    Index idx_;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version);

}; // class PtNode



/// A ParseTree is a instantiation of a AOGrammar
/// It is defined separately for the simplicity of the data structure
class ParseTree
{
public:    
    /// States of a parse tree for learning using WL-SSVM
    struct States {
        bool isBelief_;
        Scalar score_;
        Scalar loss_;
        Scalar margin_;
        Scalar norm_;

        /// Default constructor
        States();

        /// Copy constructor
        explicit States(const States & s);

        /// Constructs a state with inputs
        explicit States(bool isBelief, Scalar loss, Scalar norm);

        /// Constructs a state with inputs
        explicit States(bool isBelief, Scalar score, Scalar loss, Scalar norm);

    private:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version);

    }; // struct States

    /// Type of a detection
    typedef Detection_<Scalar> Detection;

    /// Default constructor
    ParseTree();

    /// Copy constructor
    // @note The grammar g_ can not be copied from @p pt which will be set outside constructor
    ParseTree(const ParseTree & pt);

    /// Destructor
    ~ParseTree();

    /// Constructs a parse tree for a grammar @p g
    ParseTree(const AOGrammar & g);

    /// Set the grammar for it
    void setGrammar(const AOGrammar & g);

    /// Assign operator
    ParseTree & operator=(const ParseTree & pt);

    /// Compares the scores in decreasing order
    bool operator<(const ParseTree & pt) const;

    /// Swap
    void swap(ParseTree & pt);

    /// Clears
    void clear();

    /// Returns if a parse tree is empty
    bool empty() const;

    /// Returns the nodeSet
    const std::vector<PtNode *> & nodeSet() const;
    std::vector<PtNode *> & getNodeSet();

    /// Returns the edgeSet
    const std::vector<PtEdge *> & edgeSet() const;
    std::vector<PtEdge *> & getEdgeSet();

    /// Returns the root node
    int            idxRootNode() const;
    int &          getIdxRootNode();
    const PtNode * rootNode() const;
    PtNode *& getRootNode();

    /// Returns the grammar
    const AOGrammar *  grammar() const;

    /// Returns the appearance set
    const std::vector<Appearance::Param *> & appearanceSet() const;
    std::vector<Appearance::Param *> & getAppearanceSet();

    /// Returns the bias set
    const std::vector<Scalar> &  biasSet() const;
    std::vector<Scalar> & getBiasSet();

    /// Returns the deformation set
    const std::vector<Deformation::Param *> & deformationSet() const;
    std::vector<Deformation::Param *> & getDeformationSet();

    /// Returns the Scaleprior set
    const std::vector<Scaleprior::Param *> & scalepriorSet() const;
    std::vector<Scaleprior::Param *> & getScalepriorSet();

    /// Returns the parse info set
    const std::vector<ParseInfo *> & parseInfoSet() const;
    std::vector<ParseInfo *> & getParseInfoSet();

    /// Returns the states
    const States *  states() const;
    States *& getStates();

    /// Returns dataId
    int  dataId() const;
    int & getDataId();

    /// Returns the appearance usage
    const std::vector<int> & appUsage() const;
    std::vector<int> & getAppUsage();

    /// Returns the appearance x
    const Appearance::Param *  appearaceX() const;
    Appearance::Param *& getAppearaceX();

    /// Returns image wd and ht
    int imgWd() const;
    int & getImgWd();

    int imgHt() const;
    int & getImgHt();

    /// Returns the index of object component
    int idxObjComp() const;

    /// Adds a node
    int addNode(int gNode, int type);

    /// Adds a edge
    int addEdge(int fromNode, int toNode, int gEdge);

    /// Adds a bias
    int AddBias(Scalar w);

    /// Adds a scale prior
    int addScaleprior(Scaleprior::Param & w);

    /// Adds a deformation
    int addDeformation(Scalar dx, Scalar dy, bool flip);

    /// Adds an appearance
    int addAppearance(Appearance::Param & w, bool flip);

    /// Adds a parse info
    int addParseInfo(ParseInfo & info);

    /// Visualizes it
    void showDetection(cv::Mat img, bool display=false);

    /// Returns the length of total concatenated features
    int dim() const;

    /// Compares feature values with another pt
    int compareFeatures(const ParseTree & pt) const;

    /// Computes the norm
    Scalar norm() const;

    /// Computer overlap with a given bbox
    // @param[in] ref The reference box at the original image resolution
    Scalar computeOverlapLoss(const Rectangle2i & ref) const;

    /// Visualizes the appearance
    void visualize();

    /// Finds the pt node which corresponds to the specified grammar node or its idx
    std::vector<const PtNode *> findNode(const Node * n);
    std::vector<const PtNode *> findNode(const int idxG);
          std::vector<PtNode *> getNode(const int idxG);

    /// Finds the single obj And-nodes
    /// assume the grammar model is one-layer part-based model
    std::vector<const PtNode *> findSingleObjAndNodes() const;
    std::vector<PtNode *> getSingleObjAndNodes();

    /// Gets single object detections
    void getSingleObjDet(std::vector<Detection> & dets, int ptIdx = -1);

private:
    std::vector<PtNode *>  nodeSet_;
    std::vector<PtEdge *>  edgeSet_;
    int                    idxRootNode_;

    const AOGrammar * g_;

    std::vector<Appearance::Param *> appearanceSet_;
    std::vector<Scalar>  biasSet_;
    std::vector<Deformation::Param *> deformationSet_;
    std::vector<Scaleprior::Param *> scalepriorSet_;

    std::vector<ParseInfo *> parseInfoSet_;

    int dataId_;
    States * states_;

    std::vector<int> appUsage_;

    Appearance::Param * appearanceX_; // for root 2x used in searching part configuration

    int imgWd_;
    int imgHt_;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version);

    DEFINE_RGM_LOGGER;

}; // class ParseTree



/// The example is a set of parse trees which have the same Key
class TrainExample
{
public:
    /// Type of parse tree iterator
    typedef std::vector<ParseTree>::iterator  ptIterator;

    /// Constants
    static const int NbHist = 50;

    /// Default constructor
    TrainExample();

    /// Copy constructor
    TrainExample(const TrainExample & ex);

    /// Assignment operator
    TrainExample & operator=(const TrainExample & ex);

    /// Returns parse trees
    const std::vector<ParseTree> & pts() const;
    std::vector<ParseTree> & getPts();

    /// Returns the margin bound
    Scalar   marginBound() const;
    Scalar & getMarginBound();

    /// Returns norms of belief and non-belief parse trees
    Scalar   beliefNorm() const;
    Scalar & getBeliefNorm();
    Scalar   maxNonbeliefNorm() const;
    Scalar & getMaxNonbeliefNorm();

    /// Returns the number of historical record
    int   nbHist() const;
    int & getNbHist();

    /// Checks the duplication with a parse tree
    bool isEqual(const ParseTree & pt) const;

private:
    std::vector<ParseTree> pts_;

    // For keeping track on the bound that determines if a an
    // example might possibly have a non-zero loss
    Scalar marginBound_;

    // Maximum L2 norm of the feature vectors for this example
    // (used in conjunction with margin_bound)
    Scalar beliefNorm_;
    Scalar maxNonbeliefNorm_;

    int nbHist_;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version);

}; // class TrainExample




/// Functor used to test for the intersection of two parse trees
/// according to the Pascal criterion (area of intersection over area of union).
class PtIntersector
{
public:
    /// Constructor.
    /// @param[in] reference The reference parse tree.
    /// @param[in] threshold The threshold of the criterion.
    /// @param[in] dividedByUnion Use Felzenszwalb's criterion instead (area of intersection over area
    /// of second rectangle). Useful to remove small detections inside bigger ones.
    PtIntersector(const ParseTree & reference, Scalar threshold = 0.5, bool dividedByUnion = false);

    /// Tests for the intersection between a given rectangle and the reference.
    /// @param[in] rect The rectangle to intersect with the reference.
    /// @param[out] score The score of the intersection.
    bool operator()(const ParseTree & pt, Scalar * score = 0) const;

private:
    const ParseTree * reference_;
    Scalar threshold_;
    bool   dividedByUnion_;
};

} // namespace RGM

#endif // RGM_PARSETREE_HPP_


