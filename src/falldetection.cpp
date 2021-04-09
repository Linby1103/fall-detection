//
// Created by libin on 2021/3/8.
//

#include "../include/falldetection.h"
#include <fstream>
#include <math.h>

#define MATH_PI 3.141592

detectobj::detectobj()
{
    ID=-1;
    headheight=-1.;
    xdet=-1.;
    ydet=-1.;
    aspect=0.;
    displacement-1.;
    bottom=-1;
    iou=0.;
    appearcounter=0;
}
/********************************
* @brief 计算IOU
* @param rect1，rect2 矩形框
*     -<em>false</em> -99999 or
*     -<em>true</em> iou
********************************/
static float IOU(const cv::Rect &rect1, const cv::Rect &rect2)
{
    if ((rect1.x < 0. || rect1.y < 0. || rect1.width < 0. || rect1.height < 0.) || \
		(rect2.x < 0. || rect2.y < 0. || rect2.width < 0. || rect2.height < 0.)|| \
		rect1.area()<=0. || rect2.area()<=0.)
        return -99999;

    cv::Rect union_ = rect1 | rect2;//example 7&1=1 b(7)=0111 b(1)=0001 0111 & 0001= 0001
    cv::Rect inter_ = rect1 & rect2;

    return inter_.area()*1.0 / union_.area();

}


static void Point2PointDist(const cv::Point2f &p1,const cv::Point2f &p2,double *dist)
{
    *dist=sqrt(pow(p2.x-p1.x,2)+pow(p2.y-p1.y,2));
}

static float CIOU(const cv::Rect2f &rect1, const cv::Rect2f &rect2)
{
    if ((rect1.x < 0. || rect1.y < 0. || rect1.width < 0. || rect1.height < 0.) || \
		(rect2.x < 0. || rect2.y < 0. || rect2.width < 0. || rect2.height < 0.)|| \
		rect1.area()<=0 || rect2.area()<=0)
        return -99999;

    cv::Rect union_ = rect1 | rect2;//example 7&1=1 b(7)=0111 b(1)=0001 0111 & 0001= 0001
    cv::Rect inter_ = rect1 & rect2;
    float iou=inter_.area()*1.0 / union_.area();

    double C=0.f,D=0.f;
    Point2PointDist(union_.tl(),union_.br(),&C);

    cv::Point2f cp1,cp2;
    cp1.x=(rect1.tl().x+rect1.br().x)/2;
    cp1.y=(rect1.tl().y+rect1.br().y)/2;
    cp2.x=(rect2.tl().x+rect2.br().x)/2;
    cp2.y=(rect2.tl().y+rect2.br().y)/2;
    Point2PointDist(cp1,cp2,&D);

    float det=atan2(rect2.width,rect2.height)-atan2(rect1.width,rect1.height);
    float V=4.0/pow(MATH_PI,2.0)*pow(det,2.0);
    float alph=V/(1-iou+V+0.1f);
    float ciou=iou-(D/C)-V*alph;
    return ciou;
}

void FALLDetection::init()
{
    pretarget.clear();
    curtarget.clear();

}

bool FALLDetection::point_in_rect(cv::Rect &rect, cv::Point2f &pt)
{
    if ((pt.x>rect.tl().x) && (pt.x<rect.br().x) && (pt.y>rect.tl().y) &&(pt.y<rect.br().y)) return true;

    return false;

}


int FALLDetection::feature_combination(std::vector<TrackID> &vec_currenttarget, std::vector<cv::Rect> &boxs)
{
    //1 找到与当前跟踪点对应的box
    curtarget.clear();
    for (int IDbox =0;IDbox<boxs.size();IDbox++){
        for (int IDbasept=0;IDbasept<vec_currenttarget.size();IDbasept++){
            if(point_in_rect(boxs[IDbox],vec_currenttarget[IDbasept].kp.p)){
                detectobj cur_box;
                cur_box.rect=boxs[IDbox];
                cur_box.ID=vec_currenttarget[IDbasept].ID;
                cur_box.aspect=double(cur_box.rect.width)/double(cur_box.rect.height);
                cur_box.displacement=vec_currenttarget[IDbasept].displacement;
                cur_box.headheight=cur_box.rect.br().y-vec_currenttarget[IDbasept].kp.p.y;
                curtarget.push_back(cur_box);
            }
        }
    }

    if (pretarget.size()==0)
    {                  //first frame ,init pretarget

        pretarget=curtarget;
    }

    FILE *fp = NULL;
    fp = fopen("./fall.txt", "a");

    for (int curidx=0;curidx<curtarget.size();curidx++){
        for(int preidx=0;preidx<pretarget.size();preidx++){
            if(curtarget[curidx].ID==pretarget[preidx].ID){
                curtarget[curidx].iou=CIOU(curtarget[curidx].rect,pretarget[preidx].rect);

                float xminmove1=fabs(curtarget[curidx].rect.x-pretarget[preidx].rect.x);
                float xminmove2=fabs(curtarget[curidx].rect.x+curtarget[curidx].rect.width-pretarget[preidx].rect.x-pretarget[preidx].rect.width);

                float yminmove1=fabs(curtarget[curidx].rect.y-pretarget[preidx].rect.y);
                float yminmove2=fabs(curtarget[curidx].rect.y+curtarget[curidx].rect.height-pretarget[preidx].rect.y-pretarget[preidx].rect.height);
                float xmaxdet=std::max(xminmove1,xminmove2);
                float ymaxdet=std::max(yminmove1,yminmove2);

                curtarget[curidx].ydet=ymaxdet/std::max(1.,curtarget[curidx].displacement);
                curtarget[curidx].xdet=xmaxdet/std::max(1.,curtarget[curidx].displacement);

                cv::Point2f p1(-1,-1),p2(-1,-1),p1_1(-1,-1),p2_1(-1,-1);

                p1.x=curtarget[curidx].rect.width/2+curtarget[curidx].rect.x;
                p1.y=curtarget[curidx].rect.height/2+curtarget[curidx].rect.y;

                p1_1.x=curtarget[curidx].rect.width/2+curtarget[curidx].rect.x;;
                p1_1.y=curtarget[curidx].rect.height+curtarget[curidx].rect.y;


                p2.x=pretarget[preidx].rect.x+pretarget[preidx].rect.width/2;
                p2.y=pretarget[preidx].rect.y+pretarget[preidx].rect.height;

                p2_1.x=pretarget[preidx].rect.x+pretarget[preidx].rect.width/2;;
                p2_1.y=pretarget[preidx].rect.y+pretarget[preidx].rect.height/2;

                double p1dist=0,p2dist=0;
                Point2PointDist(p1,p1_1,&p1dist);
                Point2PointDist(p2,p2_1,&p2dist);
                curtarget[curidx].bottom=(fabs(p1.x-p2.x)/fabs(p1.y-p2.y))*fabs(p1dist-p2dist);
                curtarget[curidx].appearcounter=pretarget[preidx].appearcounter;

            } else{
                curtarget[curidx].xdet=1.f;
                curtarget[curidx].ydet=1.f;
                curtarget[curidx].iou=1;
            }
        }

        std::vector<float> feat;
        feat.push_back(curtarget[curidx].displacement);
        feat.push_back(curtarget[curidx].iou);
        feat.push_back(curtarget[curidx].xdet);
        feat.push_back(curtarget[curidx].ydet);
        int result=svmclassifier.falldetection(feat,4);

        if(result<0)
        {
            curtarget[curidx].appearcounter+=1;
        }
        else
        {
            curtarget[curidx].appearcounter=0;
        }

#ifdef EXTRACT_FEATURE
        char data[256];
        if(curtarget[curidx].displacement>13.0 && curtarget[curidx].iou<0.65)
        {
            sprintf(data,"%.3f %.3f %3f %3f %3f,%d \n",curtarget[curidx].displacement,curtarget[curidx].iou,curtarget[curidx].xdet,curtarget[curidx].ydet,curtarget[curidx].bottom,-1);
            fputs(data,fp);
            printf("-----Write  success!\n");
        } else{
            sprintf(data,"%.3f %.3f %3f %3f %3f,%d \n",curtarget[curidx].displacement,curtarget[curidx].iou,curtarget[curidx].xdet,curtarget[curidx].ydet,curtarget[curidx].bottom,1);
            fputs(data,fp);
            printf("-----Write  success!-----\n");
        }
#endif
   }

    fclose(fp);
    if(curtarget.size()!=0)
        pretarget=curtarget;

    for(int i=0;i<curtarget.size();i++)
    {
        if (curtarget[i].appearcounter>=3)
            return -1;
        else if(curtarget[i].appearcounter>=2 && curtarget[i].aspect>1.2)
            return -1;
    }
    return 1e5;
}


