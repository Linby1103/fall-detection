/*
<%
setup_pybind11(cfg)
%>
*/
//#include "api.h"

#include "tracker.h"
#include <math.h>
#include "roisimilarity.h"

tracker::tracker()
{
    ;
}


int tracker::init()
{
    vec_currenttarget.clear();
    vec_newtarget.clear();
    return 0;
}


tracker::~tracker() {
    vec_currenttarget.clear();
    vec_newtarget.clear();
} // TODO

#define maxdistThreshold 200
#define losercounter 50
static void Point2PointDist(cv::Point2f &p1,cv::Point2f &p2,double *dist)
{

    *dist=sqrt(pow(p2.x-p1.x,2)+pow(p2.y-p1.y,2));
//    printf("Current coordinate :%f  %f\n",p2.x-p1.x,p2.y-p1.y);
}

void tracker::update_track(const std::vector<std::vector<cv::Rect>> &keypoints,cv::Mat &src)
{
    std::vector<TrackID> currentTrackid;
    double mindistThreshold=1e6*1.0;
    double pt2ptdistance=0.f;
    int  bestmatchidx=-1;

    double mindistance=0.f;

    static int lostalltarget=0;
    if(keypoints.size()==0)
    {
        if (lostalltarget>losercounter) {
            vec_currenttarget.clear();
            vec_newtarget.clear();
        }
        lostalltarget++;
    }

    currentTrackid.clear();
    //First frame ,we need init vec_currenttarget
    if (vec_currenttarget.size()==0)
    {
        for(int i=0;i<keypoints.size();i++)
        {
            TrackID info;
            cv::Point2f cpt;
            cpt.x=keypoints[i][0].x+(int)(keypoints[i][0].width/2);
//            cpt.y=keypoints[i][0].y+(int)(keypoints[i][0].height/2);
            cpt.y=keypoints[i][0].y+15;
            info.kp.p=cpt;
            info._rect_=keypoints[i][0];
            info.matchdata=src(keypoints[i][0]).clone();

            info.ID=i;
            info.lostcount=0;
            info.iflost= false;
            info.displacement=0;
            info.slop=1e-6;
            vec_currenttarget.push_back(info);
        }
        return;
    }
    //Init current track information
    for (int i = 0; i < keypoints.size(); i++)
    {
        TrackID _info;
        _info.ID=-1;
        cv::Point2f cpt;
        cpt.x=keypoints[i][0].x+(int)(keypoints[i][0].width/2);
//        cpt.y=keypoints[i][0].y+(int)(keypoints[i][0].height/2);
        cpt.y=keypoints[i][0].y+15;

        _info.kp.p=cpt;
        _info.lostcount=0;
        _info.iflost= false;
        _info._rect_=keypoints[i][0];
        _info.matchdata=src(keypoints[i][0]).clone();
        _info.newidappearcount=0;
        _info.slop=1e-6;
        _info.displacement=0;
        currentTrackid.push_back(_info);
    }

    //updata current vec_currenttarget
    for(int preidx=0;preidx<vec_currenttarget.size();preidx++) {

        bool update= false;
        if(vec_currenttarget[preidx].ID==-1){
            continue;
        }

        mindistThreshold=1e6*1.0;;
        bestmatchidx=-1;
        mindistance=0.f;

        for (int idx = 0; idx < currentTrackid.size(); idx++) {

            pt2ptdistance=0.f;


            Point2PointDist(vec_currenttarget[preidx].kp.p,currentTrackid[idx].kp.p,&pt2ptdistance);
            double similarity=aHash(vec_currenttarget[preidx].matchdata,currentTrackid[idx].matchdata);
//            printf("%d   %f  similarity: %.3f \n",vec_currenttarget[preidx].ID,dist,similarity);

            if ((pt2ptdistance<mindistThreshold) && (currentTrackid[idx].ID==-1 ))
            {
                //Update
                mindistThreshold=pt2ptdistance;
                if (mindistThreshold>maxdistThreshold*similarity)
                    continue;

                bestmatchidx=idx;
                mindistance=mindistThreshold;
            }
        }
        if (bestmatchidx!=-1 )//If has found best match pair
        {
            float px= fabs(currentTrackid[bestmatchidx].kp.p.x- vec_currenttarget[preidx].kp.p.x)>1e-6 ? \
            (currentTrackid[bestmatchidx].kp.p.x- vec_currenttarget[preidx].kp.p.x) : 1/(mindistance+1);

            float py=currentTrackid[bestmatchidx].kp.p.y- vec_currenttarget[preidx].kp.p.y;

            vec_currenttarget[preidx].slop=py/px;
            vec_currenttarget[preidx]._rect_=currentTrackid[bestmatchidx]._rect_;
            vec_currenttarget[preidx].matchdata=currentTrackid[bestmatchidx].matchdata.clone();
            vec_currenttarget[preidx].kp=currentTrackid[bestmatchidx].kp;
            vec_currenttarget[preidx].displacement=mindistance;
            currentTrackid[bestmatchidx].ID=vec_currenttarget[preidx].ID;
            update= true;
        }

        if(update== false)
        {
            vec_currenttarget[preidx].lostcount+=1;//Update disappeared counter
        }

        if(vec_currenttarget[preidx].lostcount>losercounter)
        {
            vec_currenttarget[preidx].iflost= true;
        }
    }

    //New target appear
    if(vec_newtarget.size()==0) {
        for (int idx = 0; idx < currentTrackid.size(); idx++) {
            if (currentTrackid[idx].ID == -1) {
                currentTrackid[idx].newidappearcount++;
                vec_newtarget.push_back(currentTrackid[idx]);//New appeared object;
            }
        }
    } else{
        for (int nid=0;nid<vec_newtarget.size();nid++){

                mindistThreshold=1e6;
                bestmatchidx=-1;

                for (int idx = 0; idx < currentTrackid.size(); idx++)
                {
                    if (currentTrackid[idx].ID!=-1)
                        continue;

                    pt2ptdistance=0;

                    Point2PointDist(vec_newtarget[nid].kp.p,currentTrackid[idx].kp.p,&pt2ptdistance);
                    double similarity=aHash(vec_newtarget[nid].matchdata,currentTrackid[idx].matchdata);

                    if ((pt2ptdistance<mindistThreshold)  && currentTrackid[idx].ID==-1)
                    {
                        //Update
                        mindistThreshold=pt2ptdistance;
                        if (mindistThreshold>maxdistThreshold*similarity)
                            continue;

                        bestmatchidx=idx;
                    }
                }

                if (bestmatchidx!=-1 )//If has found best match pair
                {
                    vec_newtarget[nid]._rect_=currentTrackid[bestmatchidx]._rect_;
                    vec_newtarget[nid].matchdata=currentTrackid[bestmatchidx].matchdata.clone();
                    vec_newtarget[nid].kp=currentTrackid[bestmatchidx].kp;

                    vec_newtarget[nid].slop=1-6;
                    vec_newtarget[nid].displacement=0; //疑似新目标不加入判断

                    currentTrackid[bestmatchidx].ID=vec_newtarget[nid].ID;

                    vec_newtarget[nid].newidappearcount++;
                } else
                {
                    vec_newtarget[nid].lostcount++;
                }

        }
    }

    //Add new track targert
    for (int idx = 0; idx < vec_newtarget.size(); idx++) {
        if(vec_newtarget[idx].lostcount>losercounter)
        {
            vec_newtarget.erase(vec_newtarget.begin()+idx);
        }

        if(vec_newtarget[idx].newidappearcount>3) {//15

            vec_newtarget[idx].ID = vec_currenttarget.size();
            vec_currenttarget.push_back(vec_newtarget[idx]);//New appeared object;
            vec_newtarget.erase(vec_newtarget.begin()+idx);
        }
    }

    //deleat lost target
    for (std::vector<TrackID>::iterator it =vec_currenttarget.begin();it<vec_currenttarget.end();it++) //Delete lose element
    {
        if(it->iflost) {
            //            vec_currenttarget.erase(it);
            it->ID = -1;
            it->kp.p=cv::Point2f (-1.0,-1.0);
            it->iflost=0;
            it->lostcount=0;
            it->_rect_=cv::Rect (-1,-1,-1,-1);
            it->matchdata=cv::Mat();
        }
    }
//    printf("Current detection target number %d   track target number %d   new target :%d \n  ",currentTrackid.size(),vec_currenttarget.size(),vec_newtarget.size());
}