#include <istream>
#include <sstream>
#include <fstream>
#include "classifier.h"
using namespace cv;
using namespace cv::ml;


static void string2float(std::string &ss,std::vector<float>& features)
{
    std::istringstream iss(ss);
    while (!iss.eof())
    {
        double feature;
        iss>>feature;
        features.push_back(feature);
    }
}

static int string2number( const std::string s)
{
    int num;
    std::stringstream ss(s);
    ss>>num;
    return num;
}

static void split(std::string str, const std::string sep,std::vector<std::string> &result)
{
    std::string::size_type pos;
    str += sep;//扩展字符串以方便操作
    int size = str.size();
    for (int i = 0; i < size; i++)
    {
        pos = str.find(sep, i);
        if (pos < size)
        {
            std::string s = str.substr(i, pos - i);
            if (s=="")
            {
                continue;
            }

            result.push_back(s);
            i = pos + sep.size() - 1;
        }
    }

}
/*************************************************
Function:       read_data_fromtxt
Description:    从txt文件中读取SVM训练数据
Calls:          trainSVM，
Input:          path txl文件路径
Output:         rawdata 训练数据，n*m,label 训练数据的类别标签
Return:         None
Others:
*************************************************/


static void read_data_fromtxt(std::string path,std::vector<std::vector<float>>& rawdata,std::vector<int>& labels)
{
    if (path==""){
        printf("File path not exist!\n");
        return;
    }
    std::ifstream datafile;
    datafile.open(path);
    if (!datafile.is_open())
    {
        printf("Can‘t open %s\n",path.c_str());
        return;
    }

    std::string ss;
    while(getline(datafile,ss))
    {
        std::vector<float> features;
        std::vector<std::string> str;

        features.clear();
        str.clear();

        split(ss,",",str);

        if(str.size()==0)
        {
            continue;
        }

        string2float(str[0],features);
        rawdata.push_back(features);
        labels.push_back(string2number(str[1]));
    }
}



/*************************************************
Function:       trainSVM
Description:    训练svm 模型，features :人体位移的速度，前后帧同一个人的ciou
Calls:          main
Input:
Output:
Return:         None
Others:
*************************************************/

void trainSVM()
{
    std::vector<int > labels;
    std::vector<std::vector<float>> trainingData;

    trainingData.clear();
    labels.clear();

    read_data_fromtxt("/home/workdir/code/HI3559Av100/ncnn_yolov5/build/fall_back.txt",trainingData,labels);
    float floatdata[trainingData.size()][3];
    int floatlabel[labels.size()];
    for (int x=0;x<trainingData.size();x++)
    {
        floatdata[x][0]=trainingData[x][0];
        floatdata[x][1]=trainingData[x][1];
        floatdata[x][2]=trainingData[x][2];
    }

    for(int y=0;y<labels.size();y++)
    {
        floatlabel[y]=labels[y];
    }

    Mat labelsMat(labels.size(), 1, CV_32SC1, floatlabel);
    Mat trainingDataMat(trainingData.size(), 2, CV_32FC1, floatdata);
    // 创建分类器并设置参数
    Ptr<SVM> model =SVM::create();
    model->setType(SVM::C_SVC);
    model->setGamma(10);
    model->setC(2);

    model->setKernel(SVM::RBF);  //核函数
    //设置训练数据
    Ptr<TrainData> tData =TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
    // 训练分类器
    model->train(tData);
    model->save("./svmtest.mat");

    trainingData.clear();
    labels.clear();
    read_data_fromtxt("/home/workdir/code/HI3559Av100/ncnn_yolov5/build/fall.txt",trainingData,labels);

    float testfloatdata[trainingData.size()][3];
    int testfloatlabel[labels.size()];
    for (int x=0;x<trainingData.size();x++)
    {
        testfloatdata[x][0]=trainingData[x][0];
        testfloatdata[x][1]=trainingData[x][1];
        testfloatdata[x][2]=trainingData[x][2];
    }

    for(int y=0;y<labels.size();y++)
    {
        testfloatlabel[y]=labels[y];
    }

    for (int i=0;i<labels.size();i++)
    {
        cv::Mat testmat(1,2,CV_32FC1,testfloatdata[i]);
        float resp=model->predict(testmat);
        printf("Predict result:%f    \t label:%d\n",resp,testfloatlabel[i]);
    }
}


/*************************************************
Function:       loadSVMmodel
Description:    加载svm模型参数
Calls:          main
Input:
Output:
Return:         None
Others:
*************************************************/
void loadSVMmodel()
{
    std::vector<int > labels;
    std::vector<std::vector<float>> trainingData;
    trainingData.clear();
    labels.clear();

    Ptr<SVM> model=cv::ml::SVM::load("/home/workdir/code/RK3399_M1808/SVM/svmtest.mat");
    read_data_fromtxt("/home/workdir/code/HI3559Av100/ncnn_yolov5/build/fall.txt",trainingData,labels);

    float testfloatdata[trainingData.size()][5];
    int testfloatlabel[labels.size()];
    for (int x=0;x<trainingData.size();x++)
    {
        testfloatdata[x][0]=trainingData[x][0];
        testfloatdata[x][1]=trainingData[x][1];
        testfloatdata[x][2]=trainingData[x][2];
        testfloatdata[x][3]=trainingData[x][3];
        testfloatdata[x][4]=trainingData[x][4];
    }

    for(int y=0;y<labels.size();y++)
    {
        testfloatlabel[y]=labels[y];
    }
    int counter=0;

    for (int i=0;i<labels.size();i++)
    {
        cv::Mat testmat(1,5,CV_32FC1,testfloatdata[i]);
        float resp=model->predict(testmat);
        if((int)(resp)== labels[i]) counter+=1;

        printf("Predict result:%f    \t label:%d\n",resp,testfloatlabel[i]);
    }
    printf("%d sample    %d current predict\n",labels.size(),counter);

}

void Fall_Classifier::init()
{
    model=cv::ml::SVM::load("/home/workdir/code/RK3399_M1808/SVM/svmtest.mat");
}
Fall_Classifier::Fall_Classifier()
{
    init();
}
/*************************************************
Function:       falldetection
Description:    检测跌倒状态
Calls:
Input:          features n维特征特征
Output:
Return:         分类结果  -1 跌倒 1 正常
Others:
*************************************************/
int Fall_Classifier::falldetection(std::vector<float>& features,int featdim)
{
    if(features.size()==0)
        return 1;
    float floatdata[featdim];
    for (int i = 0; i < features.size(); ++i)
    {
        floatdata[i]=features[i];
    }

    cv::Mat featsmat(1,featdim,CV_32FC1,floatdata);
    int res=int(model->predict(featsmat));
    return res;
}

int main12(int, char* argv[])
{
//    Fall_Classifier falldetector("/home/workdir/code/RK3399_M1808/SVM/svmtest.mat");
//    std::vector<float> feats;
//
//    std::string testdata=argv[1];
//    std::vector<float> features;
//    std::vector<std::string> str;
//
//    features.clear();
//    str.clear();
//
//    split(testdata,",",str);
//
//    if(str.size()==0)
//    {
//        return -1;
//    }
//
//    string2float(str[0],features);
//
//
//    int x=falldetector.falldetection(features,5);
//    printf("predict : %d\n",x);
    loadSVMmodel();



    return 0;

}