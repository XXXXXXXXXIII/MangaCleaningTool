#include <mct.h>
#include <bubble.h>
#include <frame.h>
#include <io.h>

#include <opencv2/ml.hpp>

#include <iostream>

using namespace mct;
using namespace std;
using namespace cv;
using namespace cv::ml;

int main()
{
    string training_data_file = "frame_property.csv";
    string algorithm_file = "frame_boost.yaml";
    Ptr<TrainData> training_data = TrainData::loadFromCSV(training_data_file, 0, 0, 1);
    Ptr<LogisticRegression> lr = LogisticRegression::create();
    lr->setLearningRate(0.001);
    lr->setIterations(100);
    lr->setRegularization(LogisticRegression::REG_L2);
    lr->setTrainMethod(LogisticRegression::BATCH);
    lr->setMiniBatchSize(1);
    //lr->train(training_data);

    Ptr<Boost> boost = Boost::create();
    boost->setBoostType(Boost::Types::GENTLE);
    boost->setWeightTrimRate(0);
    boost->train(training_data);
    //Mat param = lr->get_learnt_thetas();
    //for (int i = 0; i < param.cols; i++)
    //{
    //    cout << param.at<float>(0, i) << endl;
    //}

    boost->save(algorithm_file);
}

// String format command: 
// cat frame_boost.yaml | sed -e 's/.*/R"(&)"/' > yaml.str