#pragma once

#define NOMINMAX
#include <Windows.h>
#include <opencv2\opencv.hpp>

#include <flann/flann.hpp>
#include <boost/shared_array.hpp>

#include <math.h>
#include <atltime.h>

#define CAMERA_WIDTH 1920
#define CAMERA_HEIGHT 1080

#define DOT_SIZE 150
#define A_THRESH_VAL -5
#define DOT_THRESH_VAL_MIN 100  // ドットノイズ弾き
#define DOT_THRESH_VAL_MAX 500 // エッジノイズ弾き

#define THRESH 20

void calCoG_dot_v0(cv::Mat &src, cv::Point& sum, int& cnt, cv::Point& min, cv::Point& max, cv::Point p) 
{
	if (src.at<uchar>(p)) {
		sum += p; cnt++;
		src.at<uchar>(p) = 0;
		if (p.x<min.x) min.x = p.x;
		if (p.x>max.x) max.x = p.x;
		if (p.y<min.y) min.y = p.y;
		if (p.y>max.y) max.y = p.y;

		if (p.x - 1 >= 0) calCoG_dot_v0(src, sum, cnt, min, max, cv::Point(p.x-1, p.y));
		if (p.x + 1 < CAMERA_WIDTH) calCoG_dot_v0(src, sum, cnt, min, max, cv::Point(p.x + 1, p.y));
		if (p.y - 1 >= 0) calCoG_dot_v0(src, sum, cnt, min, max, cv::Point(p.x, p.y - 1));
		if (p.y + 1 < CAMERA_HEIGHT) calCoG_dot_v0(src, sum, cnt, min, max, cv::Point(p.x, p.y + 1));
	}
}

//ドットを検出
bool init_v0(cv::Mat &src, std::vector<cv::Point2f> &dots) 
{
	cv::Mat origSrc = src.clone();
	//適応的閾値処理
	cv::adaptiveThreshold(src, src, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 7, A_THRESH_VAL);
	cv::Mat ptsImg = cv::Mat::zeros(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3);
	cv::cvtColor(src, ptsImg, CV_GRAY2BGR);

	cv::Point sum, min, max, p;
	int cnt;
	for (int i = 0; i < CAMERA_HEIGHT; i++) {
		for (int j = 0; j < CAMERA_WIDTH; j++) {
			if (src.at<uchar>(i, j)) {
				sum = cv::Point(0, 0); cnt = 0; min = cv::Point(j, i); max = cv::Point(j, i);
				calCoG_dot_v0(src, sum, cnt, min, max, cv::Point(j, i));
				if (cnt>DOT_THRESH_VAL_MIN && max.x - min.x < DOT_THRESH_VAL_MAX && max.y - min.y < DOT_THRESH_VAL_MAX) {
					dots.push_back(cv::Point2f(sum.x / cnt, sum.y / cnt));
				}
			}
		}
	}


	std::vector<cv::Point2f>::iterator it = dots.begin();
	
	bool k = (dots.size()==DOT_SIZE);
	cv::Scalar color = k ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0);
	for (it = dots.begin(); it != dots.end(); ++it) {
		cv::circle(ptsImg, *it, 3, color, 2);
	}

	//cv::Mat resize_src, resize_pts;
	//cv::resize(origSrc, resize_src, cv::Size(), 0.5, 0.5);
	//cv::resize(ptsImg, resize_pts, cv::Size(), 0.5, 0.5);

	//cv::imshow("src", resize_src);
	//cv::imshow("result", resize_pts);

	return k;
}


//前フレームの点と現フレームの点の対応付け(currがprevの点の順に並んでくる)
void Correspond(const std::vector<cv::Point2f> &prev, const std::vector<cv::Point2f> &curr, std::vector<cv::Point2f> &track)
{
	//X: curr Y:prev

	//最近傍探索 X:カメラ点　Y:プロジェクタ点
	boost::shared_array<double> m_X ( new double [curr.size()*2] );
	for (int i = 0; i < curr.size(); i++)
	{
		m_X[i*2 + 0] = curr[i].x;
		m_X[i*2 + 1] = curr[i].y;
	}

	flann::Matrix<double> mat_X(m_X.get(), curr.size(), 2); // Xsize rows and 3 columns

	boost::shared_array<double> m_Y ( new double [prev.size()*2] );
	for (int i = 0; i < prev.size(); i++)
	{
		m_Y[i*2 + 0] = prev[i].x;
		m_Y[i*2 + 1] = prev[i].y;
	}
	flann::Matrix<double> mat_Y(m_Y.get(), prev.size(), 2); // Ysize rows and 3 columns

	flann::Index< flann::L2<double> > index( mat_X, flann::KDTreeIndexParams() );
	index.buildIndex();
			
	// find closest points
	std::vector< std::vector<size_t> > indices(prev.size());
	std::vector< std::vector<double> >  dists(prev.size());
	//indices[Yのインデックス][0] = 対応するXのインデックス
	index.knnSearch(mat_Y,
							indices,
							dists,
							1, // k of knn
							flann::SearchParams() );

	//prevの順に並び替える
	std::vector<cv::Point2f> sortedCurr;
	for(int i = 0; i < indices.size(); i++)
	{
		sortedCurr.emplace_back(curr[indices[i][0]]);
	}

	track = sortedCurr;

}

int main(int argc, char** argv)
{
    cv::VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT);
	cap.set(CV_CAP_PROP_FPS, 30);

    const int cycle = 10;
	bool refresh = true;
	cv::Scalar color;
	cv::Scalar red(0, 0, 255);
	cv::Scalar blue(255, 0, 0);

    CFileTime cTimeStart, cTimeEnd;
    CFileTimeSpan cTimeSpan;

	//初期フレーム
    cv::Mat prevFrame, prevFrameGray;
	std::vector<cv::Point2f> prevCorners;
    cap >> prevFrame;
    cv::cvtColor(prevFrame, prevFrameGray, CV_BGR2GRAY);
	init_v0(prevFrameGray, prevCorners);

    cv::waitKey(cycle);

    while (1) {
		cTimeStart = CFileTime::GetCurrentTime();           // 現在時刻

        cv::Mat frame;
        cap >> frame;
		cv::Mat drawframe = frame.clone();

        cv::Mat currFrameGray;
        std::vector<cv::Point2f> currCorners;
        std::vector<cv::Point2f> trackedCorners; //prevの順にならびかえたやつ

		//gray画像に変換
        cv::cvtColor(frame, currFrameGray, CV_BGR2GRAY);

		//ドットの検出
		init_v0(currFrameGray, currCorners);

		if(!refresh)
		{
			//前フレームとの対応付け(prevCornersをcurrCornersに対応付ける)
			Correspond(prevCorners, currCorners, trackedCorners);

			//prevと閾値以上離れてたら削除
			std::vector<cv::Point2f> trackedCorners_erase;
			for(int i = 0; i < trackedCorners.size(); i++)
			{
				if(sqrt(pow((prevCorners[i].x - trackedCorners[i].x), 2) + pow((prevCorners[i].y - trackedCorners[i].y), 2)) < THRESH)
				{
					trackedCorners_erase.emplace_back(trackedCorners[i]);
				}
			}

			std::vector<cv::Point2f> trackedCorners_new = trackedCorners_erase;
			//currCornersでまだ対応ついてない点を新規登録
			for(int i = 0; i < currCorners.size(); i++)
			{
				std::vector< cv::Point2f >::iterator cIter = find( trackedCorners_erase.begin(), trackedCorners_erase.end(), currCorners[i]);
				if( cIter == trackedCorners_erase.end())
				{
					trackedCorners_new.emplace_back(currCorners[i]);
				}
			}
			//表示
			color = refresh ? red : blue;
			for(int i = 0; i < trackedCorners_new.size(); i++)
			{
				cv::Point p = cv::Point((int) trackedCorners_new[i].x, (int) trackedCorners_new[i].y);
				cv::putText(drawframe,std::to_string(i), p, cv::FONT_HERSHEY_SIMPLEX, 0.7, color);
				cv::circle(drawframe, p, 1, color, 2);
			}
			prevCorners = trackedCorners_new;
		}
		else
		{
			prevCorners = currCorners;
		}
		int key = cv::waitKey(cycle);
		if (key == 27) break;
		else if(key == 32){ 
			refresh = !refresh;
		}

		cv::imshow("preview", drawframe);

		cTimeEnd = CFileTime::GetCurrentTime();           // 現在時刻
		cTimeSpan = cTimeEnd - cTimeStart;
		std::cout<< cTimeSpan.GetTimeSpan()/10000 << "[ms]" << std::endl;

	}


    return 0;
}