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
    // compute score maps for T-nodes   
    if ( !computeTnodeFilterResponses(pyramid) ) {
        RGM_LOG(error, "Failed to computer filter responses." );
        return false;
    }

    computeScalePriorFeature(pyramid.nbLevels());

    // Using DFS order
    std::vector<Node *> & nDFS( grammar_->getNodeDFS() );

    for ( int i = 0; i < nDFS.size(); ++i ) {
        Node * curNode = nDFS[i];
        Node::nodeType t = curNode->type();

        switch ( t ) {
        case Node::AND_NODE: {
            if ( !computeANDNode(curNode, pyramid.padx(), pyramid.pady()) ) {
                return false;
            }

            break;
        }
        case Node::OR_NODE: {
            if ( !computeORNode(curNode) ) {
                return false;
            }

            break;
        }
        } // switch t
    } // for i

    return true;
}

void Inference::runParsing(const Scalar thresh, const FeaturePyramid & pyramid,
                           Scalar & maxDetNum, std::vector<ParseTree> & pt)
{
    // Find scores above the threshold
    std::vector<Detection> cands;
    for ( int level=0; level<pyramid.nbLevels(); ++level ) {
        if ( !pyramid.validLevels()[level] ) {
            continue;
        }

        const Matrix & score( scoreMaps(grammar_->rootNode())[level] );
        const int rows = score.rows();
        const int cols = score.cols();

        for ( int y=0; y<score.rows(); ++y ) {
            for ( int x=0; x<score.cols(); ++x ) {
                const Scalar s = score(y, x);
                if ( s>thresh ) {
                    // Non-maxima suppresion in a 3x3 neighborhood
                    if (((y == 0) || (x == 0) || (s > score(y - 1, x - 1))) &&
                        ((y == 0) || (s > score(y - 1, x))) &&
                        ((y == 0) || (x == cols - 1) || (s > score(y - 1, x + 1))) &&
                        ((x == 0) || (s > score(y, x - 1))) &&
                        ((x == cols - 1) || (s > score(y, x + 1))) &&
                        ((y == rows - 1) || (x == 0) || (s > score(y + 1, x - 1))) &&
                        ((y == rows - 1) || (s > score(y + 1, x))) &&
                        ((y == rows - 1) || (x == cols - 1) || (s > score(y + 1, x + 1))))
                    {

                        cands.push_back( Detection(level, x, y, s) );	// here, (x, y) is in the coordinate with padding
                    }
                }
            }
        }
    }

    if (cands.empty()) {
        //RGM_LOG(normal, "Not found detections");
        return;
    }

    // Sort scores in descending order
    std::sort(cands.begin(), cands.end());

    if ( cands.size()>maxDetNum ) {
        cands.resize(maxDetNum);
    }

    // Compute detection windows, filter bounding boxes, and derivation trees
    int numDet = cands.size();
    pt.resize(numDet);
    Scalar mInf = -std::numeric_limits<Scalar>::infinity();

#pragma omp parallel for
    for ( int i = 0; i < numDet; ++i ) {
        parse(pyramid, cands[i], pt[i]);
        if ( param_->useNMS_ ) {
            ParseInfo * info = pt[i].getRootNode()->getParseInfo(&pt[i]);
            if ( info->clipBbox(pyramid.imgWd(), pyramid.imgHt()) ) {
                cands[i].c_ = i; // use c_ to record the index which will be used to select the pt after NMS
                cands[i].setX(info->x());
                cands[i].setY(info->y());
                cands[i].setWidth(info->width());
                cands[i].setHeight(info->height());
            } else {
                cands[i].c_ = -1;
                cands[i].score_ = mInf;
            }
        }
    }

    if ( param_->useNMS_ ) {
        std::sort(cands.begin(), cands.end());

        for (int i = 1; i < cands.size(); ++i) {
            cands.resize( std::remove_if(cands.begin() + i, cands.end(),
                                         Intersector_<Scalar>(cands[i - 1], param_->nmsOverlap_, param_->nmsDividedByUnion_)) -
                    cands.begin());
        }

        std::vector<ParseTree> ptNMS;
        ptNMS.reserve(cands.size());

        for ( int i = 0; i < cands.size(); ++i ) {
            if ( cands[i].c_ == -1 ) {
                break;
            }

            int idx = cands[i].c_;
            ptNMS.push_back( pt[idx] );
        }

        pt.swap(ptNMS);
    }
}

void Inference::parse(const FeaturePyramid & pyramid, Detection & cand,
                      ParseTree & pt)
{
    pt.clear();
    pt.setGrammar( *grammar_ );

    pt.getImgWd() = pyramid.imgWd();
    pt.getImgHt() = pyramid.imgHt();

    // Backtrack solution in BFS
    std::vector<Node *> gBFS;
    gBFS.push_back( grammar_->getRootNode() );

    // Get the parse info for the root node
    // note that cand.(x_, y_) are in the coordinate with padding
    ParseInfo pinfo(-1, cand.l_, cand.x_, cand.y_, 0, 0, 0, cand.score_, Rectangle_<Scalar>());
    int idxInfo = pt.addParseInfo( pinfo );

    // Add the root node to pt
    int t = static_cast<int>(grammar_->rootNode()->type());
    int gNode = grammar_->idxNode( grammar_->rootNode() );
    pt.getIdxRootNode() = pt.addNode(gNode, t);
    pt.getRootNode()->getIdx()[PtNode::IDX_PARSEINFO] = idxInfo;

    // BFS for pt
    std::vector<int> ptBFS;
    ptBFS.push_back( pt.idxRootNode() );

    int head = 0;
    while ( head < gBFS.size() ) {
        Node * curNode = gBFS[head];
        Node::nodeType t = curNode->type();

        switch ( t ) {
        case Node::T_NODE: {
            if ( !parseTNode(head, gBFS, ptBFS, pt) ) {
                return;
            }

            break;
        }
        case Node::AND_NODE: {
            if ( !parseANDNode(head, gBFS, ptBFS, pyramid, pt) ) {
                return;
            }

            break;
        }
        case Node::OR_NODE: {
            if ( !parseORNode(head, gBFS, ptBFS, pyramid, pt) ) {
                return;
            }

            break;
        }
        default: {
            RGM_LOG(error, "Wrong type of nodes." );
            return;
        }
        }; // switch

        head++;

    } // while
}

bool Inference::computeTnodeFilterResponses(const FeaturePyramid & pyramid)
{
    // Transform the filters if needed
#pragma omp critical
    if ( !grammar_->cachedFFTStatus() ) {
        grammar_->cachingFFTFilters();
    }

    while (!grammar_->cachedFFTStatus()) {
        RGM_LOG(normal, "Waiting for caching the FFT filters" );
    }

    // Create a patchwork
    const Patchwork patchwork(pyramid);

    // Convolve the patchwork with the filters
    int nbFilters = grammar_->cachedFFTFilters().size();
    std::vector<std::vector<Matrix> > filterResponses(nbFilters); // per Appearance per valid Level

    patchwork.convolve(grammar_->cachedFFTFilters(), filterResponses);

    if ( filterResponses.empty() ) {
        RGM_LOG(error, "filter convolution failed.");
        return false;
    }

    int nbLevel = pyramid.nbLevels();
    int nbValidLevel = filterResponses[0].size();
    assert(nbValidLevel == pyramid.nbValidLevels());

    // score maps of root node
    std::vector<Matrix>& rootScoreMaps( getScoreMaps(grammar_->rootNode()) );
    rootScoreMaps.resize(nbLevel);

    // Normalize the sizes of filter response maps per level
    for ( int l=0, ll=0; l<nbLevel; ++l ) {
        if ( pyramid.validLevels()[l] ) {
            int maxHt = 0;
            int maxWd = 0;
            for ( int i=0; i<nbFilters; ++i ) {
                maxHt = std::max<int>(maxHt, filterResponses[i][ll].rows());
                maxWd = std::max<int>(maxWd, filterResponses[i][ll].cols());
            }

            for ( int i=0; i<nbFilters; ++i ) {
                Matrix tmp = Matrix::Constant(maxHt, maxWd, -std::numeric_limits<Scalar>::infinity());
                tmp.block(0, 0, filterResponses[i][ll].rows(), filterResponses[i][ll].cols()) = filterResponses[i][ll];

                filterResponses[i][ll].swap(tmp);
            }

            rootScoreMaps[l] = Matrix::Zero(maxHt, maxWd);

            ++ll;
        } else {
            rootScoreMaps[l] = Matrix::Zero(1, 1);
        }
    }

    // Assign to T-nodes
    for ( int i=0, t=0; i<grammar_->nodeSet().size(); ++i ) {
        if ( grammar_->nodeSet()[i]->type() == Node::T_NODE ) {
            setScoreMaps( grammar_->nodeSet()[i], nbLevel, filterResponses[t], pyramid.validLevels() );
            ++t;
        }
    }

    return true;
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
    if ( node == NULL || node->type() != Node::AND_NODE ) {
        RGM_LOG(error, "Need a valid AND-node as input." );
        return false;
    }

    if ( node->outEdges().size() == 1 && node->outEdges()[0]->type() == Edge::TERMINATION ) {
        return true;
    }

    if ( node->outEdges().size() == 1 && node->outEdges()[0]->type() == Edge::DEFORMATION ) {
        // deformation rule -> apply distance transform
        Deformation::Param w = node->deformationParam();

        // init the score maps using those of the toNode
        std::vector<Matrix>& score( getScoreMaps(node) );
        score = scoreMaps(node->outEdges()[0]->toNode());

        int nbLevel = score.size();

        std::vector<MatrixXi >& x(getDeformationX(node));
        std::vector<MatrixXi >& y(getDeformationY(node));

        x.resize(nbLevel);
        y.resize(nbLevel);

        // Temporary data needed by DT, assume score[0] has the largest size
        int rows = score[0].rows();
        int cols = score[0].cols();

#pragma omp parallel for
        for ( int i=0; i<nbLevel; ++i ) {
            // Bounded distance transform with +/- 4 HOG cells (9x9 window)
            DT2D(score[i], w, Deformation::BoundedShiftInDT, x[i], y[i]);
        }

        return true;
    }

    assert(node->outEdges()[0]->type() == Edge::COMPOSITION);

    // composition rule -> shift and sum scores from toNodes
    std::vector<Matrix>& score( getScoreMaps(node) );
    score = scoreMaps(grammar_->rootNode());

    int nbLevels = score.size();

    // prepare score for this rule
    Scalar bias = 0;
    if ( node->offset() != NULL ) {
        bias = node->offset()->w() * grammar_->featureBias();
    }

    Scaleprior::Vector scalePriorScore = Scaleprior::Vector::Zero(nbLevels);
    if ( node->scaleprior() != NULL ) {
        scalePriorScore = node->scaleprior()->w() * scalepriorFeatures_;
    }

    for ( int i=0; i<nbLevels; ++i ) {
        score[i].fill(bias + scalePriorScore(i));
    }

    Scalar Inf = std::numeric_limits<Scalar>::infinity();

    // sum scores from toNodes (with appropriate shift and down sample)
    std::vector<Edge *> & outEdges = node->getOutEdges();
    for ( int i=0; i<outEdges.size(); ++i ) {
        const Node::Anchor & curAnchor = outEdges[i]->toNode()->anchor();
        int ax = curAnchor(0);
        int ay = curAnchor(1);
        int ds = curAnchor(2);

        // step size for down sampling
        int step = std::pow(2.0f, ds);

        // amount of (virtual) padding to hallucinate
        int virtpady = (step-1)*pady;
        int virtpadx = (step-1)*padx;

        // starting points (simulates additional padding at finer scales)
        // @note (ax, ay) are computed without considering padding.
        // So, given a root location (x, y) in the score map (computed with padded feature map)
        // the location of a part will be: (x-padx) * step + ax without considering padding
        // and (x-padx) * step + ax + padx = x + [ax - (step-1)*padx]
        int starty = ay-virtpady;
        int startx = ax-virtpadx;

        // score table to shift and down sample
        const std::vector<Matrix> & s( scoreMaps(outEdges[i]->toNode()) );

        for (int j = 0; j < s.size(); ++j ) {
            int level = j - grammar_->interval() * ds;
            if (level >= 0 ) {
                // ending points
                int endy = std::min<int>(s[level].rows(), starty + step*(score[j].rows()-1));
                int endx = std::min<int>(s[level].cols(), startx + step*(score[j].cols()-1));

                // y sample points
                std::vector<int> iy;
                int oy = 0;
                for ( int yy=starty; yy<endy; yy+=step ) {
                    if ( yy<0 ) {
                        oy++;
                    } else {
                        iy.push_back(yy);
                    }
                }

                // x sample points
                std::vector<int> ix;
                int ox = 0;
                for ( int xx=startx; xx<endx; xx+=step ) {
                    if ( xx<0 ) {
                        ox++;
                    } else {
                        ix.push_back(xx);
                    }
                }

                // sample scores
                Matrix sp(iy.size(), ix.size());
                for ( int yy=0; yy<iy.size(); ++yy ) {
                    for ( int xx=0; xx<ix.size(); ++xx ) {
                        sp(yy, xx) = s[level](iy[yy], ix[xx]);
                    }
                }

                // sum with correct offset
                Matrix stmp = Matrix::Constant(score[j].rows(), score[j].cols(), -Inf );
                stmp.block(oy, ox, sp.rows(), sp.cols()) = sp;
                score[j] += stmp;

            } else {
                score[j].fill( -Inf );
            }
        }
    }

    return true;
}

void Inference::DT2D(Matrix & scoreMap, Deformation::Param & w, int shift, MatrixXi & Ix, MatrixXi & Iy)
{
	// First pass of DT on columns
    Matrix colDTval = Matrix::Zero(scoreMap.rows(), scoreMap.cols());
    MatrixXi colDTidx = MatrixXi::Zero(scoreMap.rows(), scoreMap.cols());
    for ( int i=0; i < scoreMap.cols(); i++ ) {
        DT1D(scoreMap.col(i).data(),  colDTval.col(i).data(), colDTidx.col(i).data(), scoreMap.cols(), shift, scoreMap.rows(), w(2), w(3));
    }

	// Final DT
    Ix = MatrixXi::Zero(scoreMap.rows(), scoreMap.cols());
    Iy = MatrixXi::Zero(scoreMap.rows(), scoreMap.cols());
    for ( int i=0; i < scoreMap.rows(); i++) {
        DT1D(colDTval.row(i).data(), scoreMap.row(i).data(), Ix.row(i).data(), 1, shift, scoreMap.cols(), w(0), w(1));
    }

	for (int i=0; i<scoreMap.rows(); i++)
		for (int j=0; j<scoreMap.cols(); j++)
			Iy(i, j) = colDTidx(i, Ix(i, j));
}

void Inference::DT1D(const Scalar *vals, Scalar *out_vals, int *I, int step, int shift, int n, Scalar a, Scalar b)
{
    for (int i = 0; i < n; i++) {
		int left = i-shift;
		int right = i+shift;
		if (left < 0) left = 0;
		if (right > n-1) right = n-1;

		Scalar max = vals[0] - a*shift*shift + b*shift;
        int max_idx = 0;

        for (int j = left; j <= right; j++) {
            Scalar tmp = vals[j*step] - a*(i-j)*(i-j) ;
            if (tmp > max) {
                max = tmp;
                max_idx = j;
            }
        }

        out_vals[i*step] = max;
        I[i*step] = max_idx;
    }
}

bool Inference::computeORNode(Node * node)
{
    if ( node == NULL || node->type() != Node::OR_NODE ) {
        RGM_LOG(error, "Need valid OR-node as input." );
        return false;
    }

    if ( node->outEdges().size()==1 ) {
        return true;
    }

    // take pointwise max over scores of toNodes or outEdges
    std::vector<Matrix>& score( getScoreMaps(node) );
    score = scoreMaps(node->outEdges()[0]->toNode());

    for ( int i = 1; i < node->outEdges().size(); ++i ) {
        for ( int j = 0; j < score.size(); ++j ) {
            score[j] = score[j].cwiseMax(scoreMaps(node->outEdges()[i]->toNode())[j]);
        }
    } // for i

    return true;
}


bool Inference::parseORNode(int idx, std::vector<Node *> & gBFS, std::vector<int> & ptBFS,
                            const FeaturePyramid & pyramid, ParseTree & pt)
{
    Node * gNode = gBFS[idx];
    if ( gNode->type() != Node::OR_NODE ) {
        RGM_LOG(error, "Not an OR-node." );
        return false;
    }

    int fromIdx = ptBFS[idx];
    PtNode * ptNode = pt.getNodeSet()[fromIdx];

    if (ptNode->getIdx()[PtNode::IDX_PARSEINFO] == -1) {
        RGM_LOG(error, "Need parse info. for the current pt node." );
        return false;
    }

    ParseInfo * info = ptNode->getParseInfo(&pt);

    // Finds the best child of the OR-node by score matching
    int idxArgmax = -1;
    std::vector<Edge *> & outEdges( gNode->getOutEdges() );
    for ( int i = 0; i < outEdges.size(); ++i ) {
        int y = info->y_ - FeaturePyramid::VirtualPadding(pyramid.pady(), info->ds_);
        int x = info->x_ - FeaturePyramid::VirtualPadding(pyramid.padx(), info->ds_);
        Scalar s = scoreMaps(outEdges[i]->toNode())[info->l_](y, x);
        if ( info->score_ == s ) {
            idxArgmax = i;
            break;
        }
    } // for i

    if (idxArgmax == -1) {
        RGM_LOG(error, "Failed to find the best child." );
        return false;
    }

    // Get the best switching
    info->c_ = idxArgmax;
    Edge * bestEdge = outEdges[idxArgmax];
    Node * bestChild = bestEdge->getToNode();

    // Add an edge and a node to pt
    int idxG = grammar_->idxNode(bestChild);
    int t = static_cast<int>(bestChild->type());
    int toIdx = pt.addNode(idxG, t);
    PtNode * toNode = pt.getNodeSet()[toIdx];

    idxG = grammar_->idxEdge(bestEdge);
    int edge = pt.addEdge(fromIdx, toIdx, idxG);

    // Add the node to BFS
    gBFS.push_back( bestEdge->getToNode() );
    ptBFS.push_back( toIdx );

    if ( gNode == grammar_->getRootNode() ) {
        const Rectangle2i & detectWind = bestChild->detectWindow();

        // Detection scale
        Scalar scale = static_cast<Scalar>(grammar_->cellSize()) / pyramid.scales()[info->l_];

        // compute and record image coordinates of the detection window
        Scalar x1 = (info->x_ - pyramid.padx() * std::pow<int>(2, info->ds_)) * scale;
        Scalar y1 = (info->y_ - pyramid.pady() * std::pow<int>(2, info->ds_)) * scale;
        Scalar x2 = x1 + detectWind.width() * scale - 1;
        Scalar y2 = y1 + detectWind.height() * scale - 1;

        // update the parse info.
        info->setX(x1);
        info->setY(y1);
        info->setWidth(x2-x1+1);
        info->setHeight(y2-y1+1);

        // get scale prior and offset feature for toNode
        if ( bestChild->scaleprior() != NULL ) {
            Scaleprior::Param w = scalepriorFeatures_.col(info->l_);
            int idxPrior = pt.addScaleprior(w);
            toNode->getIdx()[PtNode::IDX_SCALEPRIOR] = idxPrior;
        }

        int idxBias = pt.AddBias( grammar_->featureBias() );
        toNode->getIdx()[PtNode::IDX_BIAS] = idxBias;
    }

    // pass the parse info. to the best child
    int idxInfo = pt.addParseInfo(*info); //ptNode->idx()[PtNode::IDX_PARSEINFO]; //
    toNode->getIdx()[PtNode::IDX_PARSEINFO] = idxInfo;

    return true;
}

bool Inference::parseANDNode(int idx, std::vector<Node *> & gBFS, std::vector<int> & ptBFS,
                             const FeaturePyramid & pyramid, ParseTree & pt)
{
    Node * gNode = gBFS[idx];
    if ( gNode->type() != Node::AND_NODE ) {
        RGM_LOG(error, "Not an And-node." );
        return false;
    }

    int fromIdx = ptBFS[idx];
    PtNode * ptNode = pt.getNodeSet()[fromIdx];
    if (ptNode->getIdx()[PtNode::IDX_PARSEINFO] == -1) {
        RGM_LOG(error, "Need parse info. for the current pt node." );
        return false;
    }

    ParseInfo * info = ptNode->getParseInfo(&pt);

    std::vector<Edge *> & outEdges( gNode->getOutEdges() );

    if ( outEdges.size() == 1 && outEdges[0]->type() == Edge::TERMINATION ) {

        // Add an edge and a node to pt
        int idxG = grammar_->idxNode(outEdges[0]->getToNode());
        int t = static_cast<int>(outEdges[0]->getToNode()->type());
        int toIdx = pt.addNode(idxG, t);
        PtNode * toNode = pt.getNodeSet()[toIdx];

        idxG = grammar_->idxEdge(outEdges[0]);
        int edge = pt.addEdge(fromIdx, toIdx, idxG);

        int idxInfo = pt.addParseInfo(*info); //ptNode->idx()[PtNode::IDX_PARSEINFO]; //
        toNode->getIdx()[PtNode::IDX_PARSEINFO] = idxInfo;

        // Add the node to BFS
        gBFS.push_back(outEdges[0]->getToNode());
        ptBFS.push_back(toIdx);

        return true;
    }

    if ( outEdges.size() == 1 && outEdges[0]->type() == Edge::DEFORMATION ) {

        const MatrixXi & Ix =  getDeformationX(gNode)[info->l_];
        const MatrixXi & Iy =  getDeformationY(gNode)[info->l_];

        const int vpadx = FeaturePyramid::VirtualPadding(pyramid.padx(), info->ds_);
        const int vpady = FeaturePyramid::VirtualPadding(pyramid.pady(), info->ds_);

        // Location of ptNode without virtual padding
        int nvpX = info->x_ - vpadx;
        int nvpY = info->y_ - vpady;

        // Computing the toNode's location:
        //  - the toNode is (possibly) deformed to some other location
        //  - lookup its displaced location using the distance transform's argmax tables Ix and Iy
        int defX = Ix(nvpY, nvpX);
        int defY = Iy(nvpY, nvpX);

        // with virtual padding
        int toX = defX + vpadx;
        int toY = defY + vpady;

        // get deformation vectors
        int dx = info->x_ - toX;
        int dy = info->y_ - toY;

        if (ptNode->idx()[PtNode::IDX_DEF] != -1) {
            RGM_LOG(error, "Parsing wrong deformation AND-node" );
            return false;
        }

        if (ptNode->idx()[PtNode::IDX_DEF] == -1 ) {
            ptNode->getIdx()[PtNode::IDX_DEF] = pt.addDeformation(dx, dy, gNode->isLRFlip());
        }

        // Look up the score of toNode
        const Matrix & score = scoreMaps(outEdges[0]->toNode())[info->l_];
        Scalar s = score(defY, defX);

        // Add an edge and a node to pt
        int idxG = grammar_->idxNode(outEdges[0]->getToNode());
        int t = static_cast<int>(outEdges[0]->getToNode()->type());
        int toIdx = pt.addNode(idxG, t);
        PtNode * toNode = pt.getNodeSet()[toIdx];

        idxG = grammar_->idxEdge(outEdges[0]);
        int edge = pt.addEdge(fromIdx, toIdx, idxG);

        // Detection scale and window
        Scalar scale = static_cast<Scalar>(grammar_->cellSize()) / pyramid.scales()[info->l_];
        const Rectangle2i & detectWind = outEdges[0]->toNode()->detectWindow();

        // compute and record image coordinates of the detection window
        Scalar x1 = (toX - pyramid.padx() * std::pow<int>(2, info->ds_)) * scale;
        Scalar y1 = (toY - pyramid.pady() * std::pow<int>(2, info->ds_)) * scale;
        Scalar x2 = x1 + detectWind.width()*scale - 1;
        Scalar y2 = y1 + detectWind.height()*scale - 1;

        ParseInfo pinfo(0, info->l_, toX, toY, info->ds_, dx, dy, s,
                        Rectangle_<Scalar>(x1, y1, x2-x1+1, y2-y1+1));

        toNode->getIdx()[PtNode::IDX_PARSEINFO] = pt.addParseInfo( pinfo );

        // Add the node to BFS
        gBFS.push_back(outEdges[0]->getToNode());
        ptBFS.push_back(toIdx);

        return true;
    }

    RGM_CHECK(outEdges.size() >= 1 && outEdges[0]->type() == Edge::COMPOSITION, error);
    for ( int i = 0; i < outEdges.size(); ++i ) {
        // get anchor
        const Node::Anchor & anchor = outEdges[i]->toNode()->anchor();
        int ax = anchor(0);
        int ay = anchor(1);
        int ads = anchor(2);

        // compute the location of toNode
        int toX = info->x_ * std::pow<int>(2, ads) + ax;
        int toY = info->y_ * std::pow<int>(2, ads) + ay;
        int toL = info->l_ - grammar_->interval() * ads;

        // Accumulate rescalings relative to ptNode
        int tods = info->ds_ + ads;

        // get the score of toNode
        const Matrix & score = scoreMaps(outEdges[i]->toNode())[toL];
        int nvpX = toX - FeaturePyramid::VirtualPadding(pyramid.padx(), tods);
        int nvpY = toY - FeaturePyramid::VirtualPadding(pyramid.pady(), tods);
        Scalar s = score(nvpY, nvpX);

        // Detection scale and window
        Scalar scale = static_cast<Scalar>(grammar_->cellSize()) / pyramid.scales()[toL];
        const Rectangle2i & detectWind = outEdges[i]->toNode()->detectWindow();

        // compute and record image coordinates of the detection window
        Scalar x1 = (toX - pyramid.padx()* std::pow<int>(2, tods)) * scale;
        Scalar y1 = (toY - pyramid.pady()* std::pow<int>(2, tods)) * scale;
        Scalar x2 = x1 + detectWind.width()*scale - 1;
        Scalar y2 = y1 + detectWind.height()*scale - 1;

        // Add an edge and a node to pt
        int idxG = grammar_->idxNode(outEdges[i]->getToNode());
        int t = static_cast<int>(outEdges[i]->getToNode()->type());
        int toIdx = pt.addNode(idxG, t);
        PtNode * toNode = pt.getNodeSet()[toIdx];

        idxG = grammar_->idxEdge(outEdges[i]);
        int edge = pt.addEdge(fromIdx, toIdx, idxG);

        ParseInfo pinfo(0, toL, toX, toY, tods, 0, 0, s,
                        Rectangle_<Scalar>(x1, y1, x2-x1+1, y2-y1+1));

        toNode->getIdx()[PtNode::IDX_PARSEINFO] = pt.addParseInfo(pinfo);

        // Add the node to BFS
        gBFS.push_back(outEdges[i]->getToNode());
        ptBFS.push_back(toIdx);

    } // for i

    return true;
}

bool Inference::parseTNode(int idx, std::vector<Node *> & gBFS, std::vector<int> & ptBFS,
                           ParseTree & pt)
{
    Node * gNode = gBFS[idx];
    if ( gNode->type() != Node::T_NODE ) {
        RGM_LOG(error, "Not an T-node." );
        return false;
    }

    int fromIdx = ptBFS[idx];
    PtNode * ptNode = pt.getNodeSet()[fromIdx];
    if (ptNode->getIdx()[PtNode::IDX_PARSEINFO] == -1) {
        RGM_LOG(error, "Need parse info. for the current pt node." );
        return false;
    }

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
