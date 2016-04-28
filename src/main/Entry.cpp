#include "Common.hpp"
#include "UtilLog.hpp"
#include "AOGrammar.hpp"
#include "Timer.hpp"
#include "Inference.hpp"

using namespace RGM; // Reconfigurable Grammar Model

int main(int argc, char** argv)
{
    log_init();

    DEFINE_RGM_LOGGER;
    RGM_LOG(normal, "Welcome to RGM-release 1.0");

    if (argc < 2 ) {
        RGM_LOG(warning, "Usage: ./Entry path_to_config_xml_file");
        return 0;
    }

    Timers timer;

    // read configuration
    Timers::Task * timerReadConfig = timer("ReadConfig");
    timerReadConfig->Start();

    std::string configFilename(argv[1]);
    CvFileStorage * fs = cvOpenFileStorage(configFilename.c_str(), 0,
                                           CV_STORAGE_READ);
    RGM_CHECK_NOTNULL(fs);

    std::string modelFilename = cvReadStringByName(fs, 0, "ModelFile", "");
    std::string imgFilename = cvReadStringByName(fs, 0, "ImageFile", "");
    bool visModel = cvReadIntByName(fs, 0, "VisualizeModel", 1);
    int interval = cvReadIntByName(fs, 0, "PyrInterval", 10);
    Scalar detThresh = cvReadRealByName(fs, 0, "DetectionThreshold", -0.5);
    Scalar nmsThresh = cvReadRealByName(fs, 0, "NMSThreshold", -0.5);

    cvReleaseFileStorage(&fs);

    timerReadConfig->Stop();

    // read model
    Timers::Task * timerReadModel = timer("LoadModel");
    timerReadModel->Start();

    AOGrammar g;
    RGM_CHECK(g.read(modelFilename), error);

    timerReadModel->Stop();

    // visualize model
    if (visModel) {
        Timers::Task * timerVisModel = timer("VisualizeModel");
        timerVisModel->Start();

        std::string saveDir = FileUtil::GetParentDir(modelFilename);
        g.visualize(saveDir);

        timerVisModel->Stop();
    }

    // run deteciton if a testing image is provided
    Timers::Task * timerReadImage = timer("ReadImage");
    timerReadImage->Start();

    cv::Mat img = cv::imread(imgFilename);
    if ( img.empty() ) {
        RGM_LOG(warning, "invalid test image");
        timer.showUsage();
        return 0;
    }

    timerReadImage->Stop();

    cv::imshow("TestImage (Press any key to continue)", img);
    cv::waitKey(2);

    RGM_LOG(normal, "Run Detection");

    // prepare the grammar model for detection
    // transfer filters to FFT filters for fast computing
    Timers::Task * timerInitFFTW = timer("InitFFTW");
    timerInitFFTW->Start();

    int maxHt = (img.rows + g.minCellSize() - 1) / g.minCellSize() + g.pady();
    int maxWd = (img.cols + g.minCellSize() - 1) / g.minCellSize() + g.padx();

    if (!Patchwork::InitFFTW((maxHt + 15) & ~15, (maxWd + 15) & ~15)) {
        RGM_LOG(error, "Could not initialize the Patchwork class." );
        timer.showUsage();
        return 0;
    }

    timerInitFFTW->Stop();  

    g.getInterval() = interval;
    g.getCachedFFTStatus() = false;

    // Run detection
    Timers::Task * timerRunDetection = timer("RunDetection");
    timerRunDetection->Start();

    Inference::Param inferenceParam;
    inferenceParam.useNMS_ = true;
    inferenceParam.nmsOverlap_ = nmsThresh;
    inferenceParam.nmsDividedByUnion_ = false;

    Scalar maxDetNum = 30000;
    std::vector<ParseTree> pts;

    Inference inference(g, inferenceParam); // you need to implement details of this class
    inference.runDetection(detThresh, img, maxDetNum, pts);

    timerRunDetection->Stop();

    // show detetion results
    for ( int i = 0; i < pts.size(); ++i ) {
        pts[i].showDetection(img, true);
    }

    // save results
    if ( pts.size() > 0 ) {
        std::string resultFilename = imgFilename.substr(0, imgFilename.length()-4)
                + "-" + g.name() + "-results.jpg";
        cv::imwrite(resultFilename, img);
    }

    timer.showUsage();

    return 1;
}
