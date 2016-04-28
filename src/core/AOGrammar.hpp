#ifndef RGM_AOGRAMMAR_HPP_
#define RGM_AOGRAMMAR_HPP_

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include "Parameters.hpp"
#include "Patchwork.hpp"
#include "UtilSerialization.hpp"
#include "UtilLog.hpp"


namespace RGM
{
/// Predeclaration
class Node;
class AOGrammar;
class AOGrid;

/// The Edge class represents a Rule
/// (switching, composition, deformation or termination) in the grammar
class Edge
{
public:
    /// Four types of edges:
    /// Switching   (OR-node to AND-node),
    /// Composition (AND-node to OR-node),
    /// Deformation (AND-node to TERMINAL-node / OR-node)
    /// Terminate   (AND-node to TERMINAL-node)
    enum edgeType {
        SWITCHING=0, COMPOSITION, DEFORMATION, TERMINATION, UNKNOWN_EDGE
    };

    /// Index
    enum {
        IDX_FROM = 0, IDX_TO, IDX_MIRROR, IDX_NUM
    };

    typedef Eigen::Matrix<int, 1, IDX_NUM>  Index;

    /// Constructs an empty Edge
    Edge();

    /// Copy constructor
    explicit Edge(const Edge & e);

    /// Constructs an Edge with given type @p t, @p fromNode and @p toNode
    explicit Edge(edgeType t, Node * fromNode, Node * toNode);

    /// Destructor
    ~Edge();

    /// Returns the edge type
    edgeType   type() const;
    edgeType & getType();

    /// Returns the start Node
    const Node *  fromNode() const;
    Node *& getFromNode();

    /// Returns the end Node
    const Node *  toNode() const;
    Node *& getToNode();

    /// Returns if it is mirrored Node
    bool   isLRFlip() const;
    bool & getIsLRFlip();

    /// Return left-right mirrored node
    const Edge *  lrMirrorEdge() const;
    Edge *& getLRMirrorEdge();

    /// Returns the index
    const Index & idx() const;
    Index & getIdx();

    /// Assigns the index
    void assignIdx(AOGrammar * g);

    /// Assigns the pointers to set up connections
    void assignConnections(AOGrammar * g);

private:
    void init();

private:
    edgeType edgeType_;
    Node * fromNode_;
    Node * toNode_;

    bool   isLRFlip_;
    Edge * LRMirrorEdge_;

    Index idx_;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version);

    DEFINE_RGM_LOGGER;

}; /// class Edge



/// The Node class represents a Symbol
/// (non-terminal or terminal) in the grammar
class Node
{
public:
    /// Three types of nodes:
    /// AND-node (structural decomposition),
    /// OR-node (alternative decompositions),
    /// TERMINAL-node (link to data appearance).
    enum nodeType {
        AND_NODE=0, OR_NODE, T_NODE, UNKNOWN_NODE
    };

    /// Index
    enum {
        IDX_MIRROR=0, IDX_SCALEPRIOR, IDX_BIAS, IDX_DEF, IDX_APP, IDX_FILTER,
        IDX_AOGRID, IDX_NUM
    };

    typedef Eigen::Matrix<int, 1, IDX_NUM> Index;

    /// Type of FFT filter
    typedef Patchwork::Filter FFTFilter;

    /// Type of an Anchor point in the feature pyramid  (x, y, s)
    typedef Eigen::RowVector3i  Anchor;

    /// Default constructor
    Node();

    /// Constructs an empty node of specific type
    explicit Node(nodeType t);

    /// Copy constructor
    explicit Node(const Node & n);

    /// Destructor
    ///@note All memory management are done in the AOGrammar
    ~Node();

    /// Returns node type
    nodeType   type() const;
    nodeType & getType();

    /// Returns in-edges
    const std::vector<Edge *> & inEdges() const;
    std::vector<Edge *> & getInEdges();

    /// Returns out-edges
    const std::vector<Edge *> & outEdges() const;
    std::vector<Edge *> & getOutEdges();

    /// Returns if it is mirrored Node
    bool   isLRFlip() const;
    bool & getIsLRFlip();

    /// Return left-right mirrored node
    const Node *  lrMirrorNode() const;
    Node *& getLRMirrorNode();

    /// Returns the detection window
    const Rectangle2i & detectWindow() const;
    Rectangle2i & getDetectWindow();

    /// Returns Anchor
    const Anchor & anchor() const;
    Anchor & getAnchor();

    /// Returns the scale prior feature
    const Scaleprior *  scaleprior() const;
    Scaleprior *& getScaleprior();

    /// Returns the offset
    const Offset *  offset() const;
    Offset *& getOffset();

    /// Returns deformation
    const Deformation *  deformation() const;
    Deformation *& getDeformation();

    /// Returns deformation parameters with proper flipping
    Deformation::Param deformationParam() const;

    /// Returns appearance
    const Appearance *  appearance() const;
    Appearance *& getAppearance();

    /// Returns appearance parameters with proper flipping
    Appearance::Param appearanceParam() const;

    /// Returns the FFT fiter
    const FFTFilter *  cachedFFTFilter() const;
    FFTFilter *& getCachedFFTFilter();

    /// Returns the tag
    const boost::uuids::uuid& tag() const;

    /// Returns the idx
    const std::vector<int> & idxInEdge() const;
    std::vector<int> & getIdxInEdge();

    const std::vector<int> & idxOutEdge() const;
    std::vector<int> & getIdxOutEdge();

    const Index &  idx() const;
    Index &  getIdx();

    /// Assigns the index
    void assignIdx(AOGrammar * g);

    /// Assigns the connections to the pointers
    /// @param[in] g The grammar to which it belongs to
    void assignConnections(AOGrammar * g);

private:
    /// Init
    void init();

private:
    nodeType nodeType_;
    std::vector<Edge *>  inEdges_;
    std::vector<Edge *>  outEdges_;

    bool    isLRFlip_;
    Node *  LRMirrorNode_;

    Rectangle2i   detectWindow_; // size
    Anchor        anchor_; // relative (location, scale) w.r.t. the parent node

    /// Pointers to parameters
    Scaleprior *  scaleprior_; // for subcategory AND-node
    Offset *      offset_;  // for subcategory AND-node
    Deformation * deformation_;
    Appearance * appearance_; // for T-nodes

    FFTFilter * cachedFFTFilter_;

    boost::uuids::uuid tag_; // used to index score maps in inference

    /// utility members used in save and read
    std::vector<int> idxInEdge_;
    std::vector<int> idxOutEdge_;

    Index idx_;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version);

    DEFINE_RGM_LOGGER;

}; /// class Node


/// AND-OR Grammar is embedded into an acyclic AND-OR Graph
class AOGrammar
{
public:
    /// control feature extraction
    struct FeatureExtraction {
        FeatureExtraction();
        void init();

        int    cellSize_;
        bool   extraOctave_;
        Scalar featureBias_;
        int    interval_;

    private:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version);
    };

    /// Type of a detection
    typedef Detection_<Scalar> Detection;

    /// Types of Grammar
    enum grammarType {
        STARMIXTURE=0, GRAMMAR, UNKNOWN_GRAMMAR
    };

    /// Type of regularization
    enum regType {
        REG_L2=0, REG_MAX
    };

    /// Default constructor
    AOGrammar();

    /// Destructor
    ~AOGrammar();

    /// Constructs a grammar from a saved file
    explicit AOGrammar(const std::string & modelFile);

    /// Clears the grammar
    void clear();

    /// Returns if a grammar/pt is empty
    bool empty() const;

    /// Returns type
    grammarType    type() const;
    grammarType &  getType();

    /// Returns name
    const std::string & name() const;
    std::string & getName();

    /// Returns note
    const std::string & note() const;
    std::string & getNote();

    /// Returns year
    const std::string & year() const;
    std::string & getYear();

    /// Returns the node set
    const std::vector<Node *> & nodeSet() const;
    std::vector<Node *> & getNodeSet();

    /// Return the root node
    const Node *  rootNode() const;
    Node *& getRootNode();

    /// Returns the status of having lr flip components
    bool   isLRFlip() const;
    bool & getIsLRFlip();

    /// Traces the DFS ordering of nodes
    void traceNodeDFS(Node * curNode, std::vector<int> & visited, std::vector<Node *> & nodeDFS);

    /// Returns the DFS node set
    const std::vector<Node *> & nodeDFS() const;
    std::vector<Node *> & getNodeDFS();
    const std::vector<std::vector<Node *> > & compNodeDFS() const;
    std::vector<std::vector<Node *> > & getCompNodeDFS();

    /// Traces the DFS ordering of nodes
    void traceNodeBFS(Node * curNode, std::vector<int> & visited, std::vector<Node *> & nodeBFS);

    /// Returns the DFS node set
    const std::vector<Node *> & nodeBFS() const;
    std::vector<Node *> & getNodeBFS();
    const std::vector<std::vector<Node *> > & compNodeBFS() const;
    std::vector<std::vector<Node *> > & getCompNodeBFS();

    /// Computes the component-based DFS/BFS
    void traceCompNodeDFSandBFS();

    /// Returns the edge set
    const std::vector<Edge *> & edgeSet() const;
    std::vector<Edge *> & getEdgeSet();

    /// Returns the regularization type
    regType   regMethod() const;
    regType & getRegMethod();

    /// Returns the appearance set
    const std::vector<Appearance *>  & appearanceSet() const;
    std::vector<Appearance *>  & getAppearanceSet();

    /// Returns the bias set
    const std::vector<Offset *>  & biasSet() const;
    std::vector<Offset *>  & getBiasSet();

    /// Returns the deformation set
    const std::vector<Deformation *>  & deformationSet() const;
    std::vector<Deformation *>  & getDeformationSet();

    /// Returns the scaleprior set
    const std::vector<Scaleprior *> & scalepriorSet() const;
    std::vector<Scaleprior *> & getScalepriorSet();

    /// Returns the size of max detection window
    const Rectangle2i & maxDetectWindow() const;
    Rectangle2i & getMaxDetectWindow();

    /// Returns the size of min detection window
    const Rectangle2i & minDetectWindow() const;
    Rectangle2i & getMinDetectWindow();

    /// Returns the cached FFT filters
    const std::vector<Node::FFTFilter *> &  cachedFFTFilters() const;
    std::vector<Node::FFTFilter *> &  getCachedFFTFilters();

    /// Returns the status of caching
    bool   cachedFFTStatus() const;
    bool & getCachedFFTStatus();

    /// Transfers the filters of T-nodes
    void cachingFFTFilters(bool withPCA=false);

    /// Returns the feature extraction param.
    const FeatureExtraction & featureExtraction() const;

    /// Returns the cell size used to extract features
    int   cellSize() const;
    int & getCellSize();
    int   minCellSize();

    /// Returns if extra-octave is used in feature pyramid
    bool   extraOctave() const;
    bool & getExtraOctave();

    /// Returns the feature bias
    Scalar   featureBias() const;
    Scalar & getFeatureBias();

    /// Returns the interval of pyramid
    int   interval() const;
    int & getInterval();

    /// Returns the padding
    int   padx() const;
    int   pady() const;

    /// Returns the threshold
    Scalar   thresh() const;
    Scalar & getThresh();

    /// Gets the length of total parameters
    int dim() const;

    /// Returns the index of a Node in the set of nodes
    int idxNode(const Node * node) const;

    /// Returns the index of an object And-node given a input T-node
    int idxObjAndNodeOfTermNode(const Node * tnode) const;

    /// Returns the index of an Edge in the set of edges
    int idxEdge(const Edge * edge) const;

    /// Returns the index of an Appearance in the set of appearance
    int idxAppearance(const Appearance * app) const;

    /// Returns the index of an Offset in the set of offsets
    int idxOffset(const Offset * off) const;

    /// Returns the index of a Deformation in the set of deformation
    int idxDeformation(const Deformation * def) const;

    /// Returns the index of a Scaleprior in the set of scaleprior
    int idxScaleprior(const Scaleprior * scale) const;

    /// Returns the index of a fft filter
    int idxFFTFilter(const Node::FFTFilter *  filter) const;

    /// Returns the Node with the given index
    Node * findNode(int idx);

    /// Returns the Edge with the given index
    Edge * findEdge(int idx);

    /// Returns the Appearance with the given index
    Appearance * findAppearance(int idx);

    /// Returns the Offset with the given index
    Offset * findOffset(int idx);

    /// Returns the Deformation with the given index
    Deformation * findDeformation(int idx);

    /// Returns the scaleprior with the given index
    Scaleprior * findScaleprior(int idx);

    /// Returns the fft filter with the given index
    Node::FFTFilter *  findCachedFFTFilter(int idx);

    /// Saves to a stream
    /// @param[in] archiveType 0 binary, 1 text
    void save(const std::string & modelFile, int archiveType = 0);

    /// Reads from a stream
    bool read(const std::string & modelFile, int archiveType = 0);

    /// Visualize the grammar using GraphViz
    void visualize(const std::string & saveDir);

    /// Adds a node
    Node * addNode(Node::nodeType t);

    /// Adds an edge
    Edge * addEdge(Node * from, Node * to, Edge::edgeType t);

    /// Adds a bias
    Offset * addOffset(const Offset & bias);

    /// Adds a scale prior
    Scaleprior * addScaleprior(const Scaleprior & prior);

    /// Adds a deformation
    Deformation * addDeformation(const Deformation & def);

    /// Adds a child node
    std::pair<Node *, Edge *> addChild(Node * parent, Node::nodeType chType, Edge::edgeType edgeType);

    /// Adds a parent node
    std::pair<Node *, Edge *> addParent(Node * ch, Node::nodeType paType, Edge::edgeType edgeType);

    /// Adds a mirrored copy of a subgraph rooted at a given node
    Node * addLRMirror(Node * root);

    /// Links two nodes as a left-right pair
    void setNodeLRFlip(Node * n, Node * nFlip);

    /// Links two edges as a left-right pair
    void setEdgeLRFlip(Edge * e, Edge * eFlip);

    /// Finalizes the grammar
    void finalize(bool hasIdx);

private:
    /// Init
    void init();

    /// Visualizes appearance templates of T-nodes
    void pictureTNodes(const std::string & saveDir);

    /// Visualizes deformation
    void pictureDeformation(const std::string & saveDir);

private:
    grammarType gType_;
    std::string name_; // class name, e.g., person
    std::string note_; // any notes, e.g., trained on PASCAL VOC
    std::string year_; // e.g., 2007 (indicating what year of VOC)

    std::vector<Node *>  nodeSet_;  // Maintaining the graph structure
    std::vector<Edge *>  edgeSet_;
    Node *               rootNode_;
    bool                 isLRFlip_;

    std::vector<Node *>  nodeDFS_; // Depth-First-Search ordering of nodes
    std::vector<Node *>  nodeBFS_; // Breadth-First-Search ordering of nodes

    std::vector<std::vector<Node *> >  compNodeDFS_; // Depth-First-Search ordering of nodes for each component
    std::vector<std::vector<Node *> >  compNodeBFS_; // Breadth-First-Search ordering of nodes

    regType regMethod_;
    std::vector<Appearance *>  appearanceSet_; // All parameters
    std::vector<Offset *> biasSet_;
    std::vector<Deformation *> deformationSet_;
    std::vector<Scaleprior *> scalepriorSet_;

    Rectangle2i maxDetectWindow_;
    Rectangle2i minDetectWindow_;

    FeatureExtraction  featureExtraction_;

    Scalar thresh_;    

    /// computing filter responses with FFT
    std::vector<Node::FFTFilter *>  cachedFFTFilters_;
    bool cached_;

    /// utility members used in save and read
    int idxRootNode_;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version);

    DEFINE_RGM_LOGGER;

}; /// class AOGrammar


/// Serialize a vector of models
void save(const std::string & modelFile, const std::vector<AOGrammar> & models, int archiveType = 0);
bool load(const std::string & modelFile, std::vector<AOGrammar> & models, int archiveType = 0);

} // namespace RGM

/// Somehow, I can not serialize a vector of AOGrammar without the following codes
namespace boost
{
namespace serialization
{
template <class Archive>
void save(Archive & ar, const std::vector<RGM::AOGrammar> & models, const unsigned int version)
{
    ar.register_type(static_cast<RGM::AOGrammar *>(NULL));
    ar.template register_type<RGM::AOGrammar>();

    int num = models.size();

    ar & BOOST_SERIALIZATION_NVP(num);

    for ( int i = 0;  i < num; ++i ) {
        ar & models[i];
    }

}

template <class Archive>
void load(Archive & ar, std::vector<RGM::AOGrammar> & models, const unsigned int version)
{
    ar.register_type(static_cast<RGM::AOGrammar *>(NULL));
    ar.template register_type<RGM::AOGrammar>();

    int num;

    ar & BOOST_SERIALIZATION_NVP(num);

    models.resize(num);
    for ( int i = 0;  i < num; ++i ) {
        ar & models[i];
    }
}

template <class Archive>
void serialize(Archive & ar, std::vector<RGM::AOGrammar> & models, const unsigned int version)
{
    split_free(ar,models,version);
}

} // namespace serialization
} // namespace boost

#endif // RGM_AOGRAMMAR_HPP_
