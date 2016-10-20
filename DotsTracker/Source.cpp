#pragma once

#define NOMINMAX
#include <Windows.h>
#include <opencv2\opencv.hpp>

#include <flann/flann.hpp>
#include <boost/shared_array.hpp>

#include <atltime.h>

#define CAMERA_WIDTH 1920
#define CAMERA_HEIGHT 1080

#define DOT_SIZE 150
#define A_THRESH_VAL -5
#define DOT_THRESH_VAL_MIN 100  // �h�b�g�m�C�Y�e��
#define DOT_THRESH_VAL_MAX 500 // �G�b�W�m�C�Y�e��

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

//�h�b�g�����o
bool init_v0(cv::Mat &src, std::vector<cv::Point2f> &dots) 
{
	cv::Mat origSrc = src.clone();
	//�K���I臒l����
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

	cv::Mat resize_src, resize_pts;
	cv::resize(origSrc, resize_src, cv::Size(), 0.5, 0.5);
	cv::resize(ptsImg, resize_pts, cv::Size(), 0.5, 0.5);

	cv::imshow("src", resize_src);
	cv::imshow("result", resize_pts);

	return k;
}


//�O�t���[���̓_�ƌ��t���[���̓_�̑Ή��t��(curr��prev�̓_�̏��ɕ���ł���)
void Correspond(const std::vector<cv::Point2f> &prev, std::vector<cv::Point2f> &curr)
{
	//X: curr Y:prev

	//�ŋߖT�T�� X:�J�����_�@Y:�v���W�F�N�^�_
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
	//indices[Y�̃C���f�b�N�X][0] = �Ή�����X�̃C���f�b�N�X
	index.knnSearch(mat_Y,
							indices,
							dists,
							1, // k of knn
							flann::SearchParams() );

	//prev�̏��ɕ��ёւ���
	std::vector<cv::Point2f> sortedCurr;
	for(int i = 0; i < indices.size(); i++)
	{
		sortedCurr.emplace_back(curr[indices[i][0]]);
	}

	curr = sortedCurr;

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

	//�����t���[��
    cv::Mat prevFrame, prevFrameGray;
	std::vector<cv::Point2f> prevCorners;
    cap >> prevFrame;
    cv::cvtColor(prevFrame, prevFrameGray, CV_BGR2GRAY);
	init_v0(prevFrameGray, prevCorners);

    cv::waitKey(cycle);

    while (1) {
		cTimeStart = CFileTime::GetCurrentTime();           // ���ݎ���

        cv::Mat frame;
        cap >> frame;
		cv::Mat drawframe = frame.clone();

        cv::Mat currFrameGray;
        std::vector<cv::Point2f> currCorners;

		//gray�摜�ɕϊ�
        cv::cvtColor(frame, currFrameGray, CV_BGR2GRAY);

		//�h�b�g�̌��o
		init_v0(currFrameGray, currCorners);

		if(!refresh)
		{
			//�O�t���[���Ƃ̑Ή��t��(prevCorners��currCorners�ɑΉ��t����)
			Correspond(prevCorners, currCorners);
		}
		//�\��
		color = refresh ? red : blue;
		for(int i = 0; i < currCorners.size(); i++)
		{
			cv::Point p = cv::Point((int) currCorners[i].x, (int) currCorners[i].y);
			cv::putText(drawframe,std::to_string(i), p, cv::FONT_HERSHEY_SIMPLEX, 0.7, color);
			cv::circle(drawframe, p, 1, color, 2);
		}
		cv::imshow("preview", drawframe);

		int key = cv::waitKey(cycle);
		if (key == 27) break;
		else if(key == 32){
			refresh = !refresh;
		}

		prevCorners = currCorners;

//�R�[�i�[�_�̌��o�ƃg���b�L���O
#if 0
        // �����_���o

        std::vector<uchar> featuresFound;
        std::vector<float> featuresErrors;

		if(refresh){
			//�h�b�g���o
		}else{
		    cTimeStart = CFileTime::GetCurrentTime();           // ���ݎ���
			cv::calcOpticalFlowPyrLK(
				prevFrameGray,
				currFrameGray,
				prevCorners,
				currCorners,
				featuresFound,
				featuresErrors);
			cTimeEnd = CFileTime::GetCurrentTime();           // ���ݎ���
			cTimeSpan = cTimeEnd - cTimeStart;
			std::cout<< cTimeSpan.GetTimeSpan()/10000 << "[ms]" << std::endl;
		}
        for (int i = 0; i < currCorners.size(); i++) {
            cv::Point p2 = cv::Point((int) currCorners[i].x, (int) currCorners[i].y);
			if(refresh)	{
				cv::putText(drawframe,std::to_string(i), cv::Point((int) currCorners[i].x, (int) currCorners[i].y), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255));
				cv::circle(drawframe, p2, 1, cv::Scalar(0,0,255), 2);
			}else{
				if(featuresErrors[i] <= 30.0f) //��ʊO�ɔ�яo�Ă�
				{
				cv::putText(drawframe,std::to_string(i), cv::Point((int) currCorners[i].x, (int) currCorners[i].y), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,0,0));
				cv::circle(drawframe, p2, 3, cv::Scalar(255,0,0), 2);

				}
			}
        }


		//�R�[�i�[���o���ʕ\��
		cv::Mat resize_cam;
		cv::resize(drawframe, resize_cam, cv::Size(), 0.8, 0.8);
		cv::imshow("preview", drawframe);
        prevFrame = frame;
		prevCorners = currCorners;

		int key = cv::waitKey(cycle);

		if (key == 27) break;
		else if(key == 32){
			refresh = !refresh;
		}
#endif

#if 0

        // �����_���o
        std::vector<cv::Point2f> prevCorners;
        std::vector<cv::Point2f> currCorners;

        cv::goodFeaturesToTrack(prevFrameGray, prevCorners, 20, 0.05, 5.0);
        cv::goodFeaturesToTrack(currFrameGray, currCorners, 20, 0.05, 5.0);
        cv::cornerSubPix(prevFrameGray, prevCorners, cv::Size(21, 21), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01));
        cv::cornerSubPix(currFrameGray, currCorners, cv::Size(21, 21), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01));

        std::vector<uchar> featuresFound;
        std::vector<float> featuresErrors;

        cv::calcOpticalFlowPyrLK(
            prevFrameGray,
            currFrameGray,
            prevCorners,
            currCorners,
            featuresFound,
            featuresErrors);

        for (int i = 0; i < featuresFound.size(); i++) {
            cv::Point p1 = cv::Point((int) prevCorners[i].x, (int) prevCorners[i].y);
            cv::Point p2 = cv::Point((int) currCorners[i].x, (int) currCorners[i].y);

			//cv::putText(drawframe,std::to_string(i), cv::Point((int) currCorners[i].x, (int) currCorners[i].y), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,0,0));
			//cv::circle(drawframe, p2, 3, cv::Scalar(255,0,0), 2);
            cv::line(drawframe, p1, p2, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("preview", drawframe);
        prevFrame = frame;
        if (cv::waitKey(cycle) == 27) { break; }
#endif
		cTimeEnd = CFileTime::GetCurrentTime();           // ���ݎ���
		cTimeSpan = cTimeEnd - cTimeStart;
		std::cout<< cTimeSpan.GetTimeSpan()/10000 << "[ms]" << std::endl;

	}


    return 0;
}