#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <windows.h>
#include <tchar.h> 
#include <stdio.h>
#include <strsafe.h>
#include <time.h>
#include <list>
 
#define PI 3.14159265
 
using namespace cv;
using namespace std;
 
Ptr<FaceRecognizer> model;
vector<Mat> images;
vector<int> labelints;
vector<string> labels;
/**
** CONFIGURATION SECTION
**/
int method = 1; // 1 eigenfaces, 2 fisherfaces, 3 HSV 
bool info = true; // Print info in console
bool rotation_def = true; // enable/disable face rotation
const int treshold_slider_max = 100000; // maximum treshold
int treshold_slider = 15000; // default treshold for face recognition
int logCountMax = 50; //each x frame will be logged
bool fixingData = false; // fixing training images
bool noseDetection = true; // rotation according to nose position
/**
*************************
**/
std::list<int> noseQueue;
double treshold = treshold_slider;
int logCount = 0;
void detectAndDisplay(Mat frame, bool &capture, string user, bool recognize);
void noseposition(Mat frame);
void learnFacesEigen();
void learnFacesFisher();
float gradienty(Mat frame);
void learnFacesLBPH();
void getTrainData();
void fixData();
Mat cropFace(Mat srcImg, int eyeLeftX, int eyeLeftY, int eyeRightX, int eyeRightY, int width, int height, int faceX, int faceY, int faceWidth, int faceHeight);
 
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eye_cascade_name = "haarcascade_eye.xml";
 
CascadeClassifier face_cascade;
CascadeClassifier nose_cascade;
CascadeClassifier eye_cascade;
string window_name = "Capture - Face detection";
 
ofstream logFile;
 
void on_trackbar(int, void*)
{
    treshold = treshold_slider;
}
void callbackButton(int, void*)
{
     
}
 
static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch (src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}
 
int main(int argc, const char** argv)
{
 
    Mat image;
    Mat frame;
 
    logFile = ofstream("logs.csv", std::ios_base::app);
    if (!logFile.is_open())
    {
        cout << "Cannot open log file";
        return 0;
    }
 
 
    //-- 1. Load the cascades
    if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading\n"); return -1; };
    if (!eye_cascade.load(eye_cascade_name)){ printf("--(!)Error loading\n"); return -1; };
    // Get train data
    getTrainData();
 
    if (labels.size() > 0)
    {
        //learn faces
        if (method == 1)
        {
            learnFacesEigen();
 
        }
        else if
            (method == 2){
            learnFacesFisher();
        }
        else{
            learnFacesLBPH();
        }
    }
 
    VideoCapture stream1(1);
    //-- 2. Read the video stream
    if (!stream1.isOpened()){
        cout << "cannot open camera";
    }
 
    Sleep(2000);
    bool capture = false;
    bool recognize = false;
    /*
    ** GUI
    */
    namedWindow(window_name, 1);
 
    char TrackbarName[50];
    sprintf(TrackbarName, "Treshold %d", treshold_slider_max);
    // trackbar for treshold
    createTrackbar(TrackbarName, window_name, &treshold_slider, treshold_slider_max, on_trackbar);
    string user;
    /*fixingData = true;
    fixData();
    fixingData = false;*/
    while (true)
    {
        bool test = stream1.read(frame);
 
        //-- 3. Apply the classifier to the frame
        if (test)
        {
            detectAndDisplay(frame, capture, user, recognize);
        }
        else
        {
            printf(" --(!) No captured frame -- Break!"); break;
        }
 
        int c = waitKey(10);
        if ((char)c == 'n'){
            user.clear();
            cin >> user;
        }
        // press r for face recognition
        if ((char)c == 'r'){ recognize = !recognize; }
        // press a for face capture
        if ((char)c == 'a'){ capture = true; }
        // press e to print out eigenfaces
        if ((char)c == 'e') { info = false; }
        //press c to exit
        if ((char)c == 'c') { break; }
    }
 
    stream1.release();
    logFile.close();
    waitKey(0);
    return 0;
}
 
 
void detectAndDisplay(Mat frame, bool &capture, string user, bool recognize)
{
    std::vector<Rect> faces;
    std::vector<Rect> hss;
    Mat frame_gray;
    Mat croppedImage;
    Mat workingImage;
    int fontFace = FONT_HERSHEY_PLAIN;
    double fontScale = 2;
    int thickness = 1;
    if (!fixingData)
    {
 
        cvtColor(frame, frame_gray, CV_BGR2GRAY);
        //equalizeHist(frame_gray, frame_gray);
    }
    else
    {
 
        frame_gray = frame;
    }
 
    //-- Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING, Size(frame.size().width*0.2, frame.size().height*0.2));
 
    for (size_t i = 0; i < faces.size(); i++)
    {
 
        int tlY = faces[i].y;
        //tlY -= (faces[i].height / 3);
        if (tlY < 0){
            tlY = 0;
        }
        int drY = faces[i].y + faces[i].height;
        //drY += +(faces[i].height / 6);
        if (drY>frame.rows)
        {
            drY = frame.rows;
        }
        Point tl(faces[i].x, tlY);
        Point dr(faces[i].x + faces[i].width, drY);
 
        Rect myROI(tl, dr);
        Mat croppedImage_original = frame(myROI);
        Mat croppedImageGray;
 
 
        //resize(croppedImage_original, croppedImage_original, Size(200, 292), 0, 0, INTER_CUBIC);
        if (!fixingData)
        {
            cvtColor(croppedImage_original, croppedImageGray, CV_RGB2GRAY);
        }
        else
        {
            croppedImageGray = croppedImage_original;
        }
 
        std::vector<Rect> eyes;
        // detect eyes
        eye_cascade.detectMultiScale(croppedImageGray, eyes, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING, Size(croppedImageGray.size().width*0.2, croppedImageGray.size().height*0.2));
 
        int eyeLeftX = 0;
        int eyeLeftY = 0;
        int eyeRightX = 0;
        int eyeRightY = 0;
        for (size_t f = 0; f < eyes.size(); f++)
        {
            int tlY2 = eyes[f].y + faces[i].y;
            if (tlY2 < 0){
                tlY2 = 0;
            }
            int drY2 = eyes[f].y + eyes[f].height + faces[i].y;
            if (drY2>frame.rows)
            {
                drY2 = frame.rows;
            }
            Point tl2(eyes[f].x + faces[i].x, tlY2);
            Point dr2(eyes[f].x + eyes[f].width + faces[i].x, drY2);
 
            if (eyeLeftX == 0)
            {
 
                //rectangle(frame, tl2, dr2, Scalar(255, 0, 0));
                eyeLeftX = eyes[f].x;
                eyeLeftY = eyes[f].y;
            }
            else if (eyeRightX == 0)
            {
 
                ////rectangle(frame, tl2, dr2, Scalar(255, 0, 0));
                eyeRightX = eyes[f].x;
                eyeRightY = eyes[f].y;
 
            }
 
        }
        // if lefteye is lower than right eye swap them
        if (eyeLeftX > eyeRightX){
            croppedImage = cropFace(frame_gray, eyeRightX, eyeRightY, eyeLeftX, eyeLeftY, 200, 200, faces[i].x, faces[i].y, faces[i].width, faces[i].height);
        }
        else{
            croppedImage = cropFace(frame_gray, eyeLeftX, eyeLeftY, eyeRightX, eyeRightY, 200, 200, faces[i].x, faces[i].y, faces[i].width, faces[i].height);
        }
 
 
        if (capture && !user.empty())
        {
            // save captured image to file
            string result;
            unsigned long int sec = time(NULL);
            result << "facedata/" << user << "_" << sec << ".jpg";
            imwrite(result, croppedImage);
            capture = false;
        }
        if (recognize&&!croppedImage.empty())
        {
            //recognize face
            logCount++;
            int predictedLabel = -1;
            double predicted_confidence = 0.0;
            model->set("threshold", treshold);
            model->predict(croppedImage, predictedLabel, predicted_confidence);
            string text;
            if (logCount > logCountMax)
            {
                if (predictedLabel > -1)
                {
                    // log to file
                    text = labels[predictedLabel];
                    logCount = 0;
                    logFile << text;
                    logFile << ";";
                    logFile << predicted_confidence;
                    logFile << "\n";
                }
            }
            cout << predicted_confidence;
            cout << "\n";
            if (!info&&method == 1)
            {
                // save Eigenvectors to file
                info = true;
                // Here is how to get the eigenvalues of this Eigenfaces model:
                Mat eigenvalues = model->getMat("eigenvalues");
                // And we can do the same to display the Eigenvectors (read Eigenfaces):
                Mat W = model->getMat("eigenvectors");
                // Get the sample mean from the training data
                Mat mean = model->getMat("mean");
                // Display or save:
                imwrite(format("%s/mean.png", "eigenfaces"), norm_0_255(mean.reshape(1, 292)));
                // Display or save the Eigenfaces:
                for (int i = 0; i < min(10, W.cols); i++) {
                    string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
                    cout << msg << endl;
                    // get eigenvector #i
                    Mat ev = W.col(i).clone();
                    // Reshape to original size & normalize to [0...255] for imshow.
                    Mat grayscale = norm_0_255(ev.reshape(1, 292));
                    // Show the image & apply a Jet colormap for better sensing.
                    Mat cgrayscale;
                    applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
                    // Display or save:
 
                    imwrite(format("%s/eigenface_%d.png", "eigenfaces", i), norm_0_255(cgrayscale));
 
                }
            }
 
            // display name of recognized person
            if (predictedLabel > -1)
            {
 
                text = labels[predictedLabel];
                putText(frame, text, tl, fontFace, fontScale, Scalar::all(255), thickness, 8);
            }
 
        }
        rectangle(frame, tl, dr, Scalar(255, 0, 255));
 
    }
 
    imshow(window_name, frame);
}
std::wstring s2ws(const std::string& s)
{
    int len;
    int slength = (int)s.length() + 1;
    len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
    wchar_t* buf = new wchar_t[len];
    MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
    std::wstring r(buf);
    delete[] buf;
    return r;
}
 
vector<string> get_all_files_names_within_folder(wstring folder)
{
    vector<string> names;
    TCHAR search_path[200];
    StringCchCopy(search_path, MAX_PATH, folder.c_str());
    StringCchCat(search_path, MAX_PATH, TEXT("\\*"));
    WIN32_FIND_DATA fd;
    HANDLE hFind = ::FindFirstFile(search_path, &fd);
    if (hFind != INVALID_HANDLE_VALUE)
    {
        do
        {
            // read all (real) files in current folder
            // , delete '!' read other 2 default folder . and ..
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
            {
                //_tprintf(TEXT("  %s   <DIR>\n"), fd.cFileName);
                wstring test = fd.cFileName;
                string str(test.begin(), test.end());
                names.push_back(str);
            }
        } while (::FindNextFile(hFind, &fd));
        ::FindClose(hFind);
    }
    return names;
}
void getTrainData(){
    wstring folder(L"facedata/");
    vector<string> files = get_all_files_names_within_folder(folder);
    string fold = "facedata/";
    int i = 0;
    for (std::vector<string>::iterator it = files.begin(); it != files.end(); ++it) {
        images.push_back(imread(fold + *it, 0));
        labelints.push_back(i);
        string str = *it;
        unsigned pos = str.find("_");
        string str2 = str.substr(0, pos);
        labels.push_back(str2);
        i++;
    }
}
void fixData(){
    Mat fixImg;
    wstring folder(L"facedata2/");
    vector<string> files = get_all_files_names_within_folder(folder);
    string fold = "facedata2/";
    for (std::vector<string>::iterator it = files.begin(); it != files.end(); ++it) {
        fixImg = imread(fold + *it, 0);
        string str = *it;
        unsigned pos = str.find("_");
        string str2 = str.substr(0, pos);
        bool capture = true;
        detectAndDisplay(fixImg, capture, str2, true);
        waitKey(1000);
    }
}
void learnFacesEigen(){
 
    model = createEigenFaceRecognizer();
    model->train(images, labelints);
}
void learnFacesFisher(){
    model = createFisherFaceRecognizer();
    model->train(images, labelints);
}
void learnFacesLBPH(){
    model = createLBPHFaceRecognizer();
    model->train(images, labelints);
}
void rotate(cv::Mat& src, double angle, cv::Mat& dst)
{
    int len = max(src.cols, src.rows);
    cv::Point2f pt(len / 2., len / 2.);
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
 
    cv::warpAffine(src, dst, r, cv::Size(len, len));
}
 
int plotHistogram(Mat image)
{
    Mat dst;
 
    /// Establish the number of bins
    int histSize = 256;
 
    /// Set the ranges
    float range[] = { 0, 256 };
    const float* histRange = { range };
 
    bool uniform = true; bool accumulate = false;
 
    Mat b_hist, g_hist, r_hist;
    /// Compute the histograms:
    calcHist(&image, 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
 
    int hist_w = 750; int hist_h = 500;
    int bin_w = cvRound((double)hist_w / histSize);
 
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
 
    /// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    int sum = 0;
    int max = 0;
    int now;
    int current = 0;
    for (int i = 1; i < histSize; i++)
    {
        /*line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
            Scalar(255, 255, 255), 2, 8, 0);*/
        now = cvRound(b_hist.at<float>(i));
        // ak su uhly v rozsahu 350-360 alebo 0-10 dame ich do suctu
        if ((i < 5))
        {
            max += now;
            current = i;
        }
        /*if ((i > 175))
        {
            max += now;
            current = i;
        }*/
 
    }
 
    return max;
 
}
float gradienty(Mat frame)
{
 
    Mat src, src_gray;
    int scale = 1;
    int delta = 0;
    src_gray = frame;
    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Mat magnitudes, angles;
    Mat bin;
    Mat rotated;
    int max = 0;
    int uhol = 0;
    // otocime obrazok do uhlov v ktorych je este mozne tvar detekovat
    for (int i = -50; i < 50; i++)
    {
        rotate(src_gray, ((double)i / PI), rotated);
        // aplikujeme vertikalny a horizontalny sobel
        Sobel(rotated, grad_x, CV_32F, 1, 0, 9, scale, delta, BORDER_DEFAULT);
        Sobel(rotated, grad_y, CV_32F, 0, 1, 9, scale, delta, BORDER_DEFAULT);
        // vypocitame uhly
        cartToPolar(grad_x, grad_y, magnitudes, angles);
        // skonvertujeme do stupnov sedej pre lepsiu reprezentaciu 1 stupenm sedej = 2 stupne (este mozno zmenit!!!)
        angles.convertTo(bin, CV_8U, 90 / PI);
        // vyrezeme nos ktory sa nachadza v strede hlavy +-
        Point tl((bin.cols / 2) - 10, (bin.rows / 2) - 20);
        Point dr((bin.cols / 2) + 10, (bin.rows / 2));
        Rect myROI(tl, dr);
        Mat working_pasik = bin(myROI);
        int current = 0;
        // vypocitame histogram a pocet uhlov v norme
        current = plotHistogram(working_pasik);
        // vyberieme maximum
        if (current > max)
        {
            max = current;
            uhol = i;
        }
    }
    //pridame uhol do queue
    noseQueue.push_back(uhol);
    int suma = 0;
    // spocitame vsetky uhly v queue
    for (std::list<int>::iterator it = noseQueue.begin(); it != noseQueue.end(); it++)
    {
 
        suma = suma + *it;
    }
    //cout << "suma " << suma << "\n";
    int priemer;
    // vypocitame priemerny uhol za posledne 4 snimky
    priemer = (int)((double)suma / (double)noseQueue.size());
    // ak je moc v queue posledne odstranime
    if (noseQueue.size() > 3)
    {
        noseQueue.pop_front();
    }
    // vypiseme priemer
    cout << priemer;
    cout << "\n";
 
    return priemer;
 
 
}
 
Mat cropFace(Mat srcImg, int eyeLeftX, int eyeLeftY, int eyeRightX, int eyeRightY, int width, int height, int faceX, int faceY, int faceWidth, int faceHeight){
    Mat dstImg;
    Mat crop;
    // Ak sa detekuju oci, otacaj podla vyskoveho rozdielu oci
    if (!(eyeLeftX == 0 && eyeLeftY == 0))
    {
 
        int eye_directionX = eyeRightX - eyeLeftX;
        int eye_directionY = eyeRightY - eyeLeftY;
        float rotation = atan2((float)eye_directionY, (float)eye_directionX) * 180 / PI;
        if (rotation_def){
            rotate(srcImg, rotation, dstImg);
        }
        else {
            dstImg = srcImg;
        }
    } // Ak sa oci nedetekuju a je zapnuta detekcia nosa otacaj podla histogramu orientovanych gradientov nosa
    else
    {
 
        if (noseDetection)
        {
            Point tl(faceX, faceY);
            Point dr((faceX + faceWidth), (faceY + faceHeight));
 
            Rect myROI(tl, dr);
            Mat croppedImage_original = srcImg(myROI);
 
            Mat noseposition_image;
            //Zmensime oblast orezanim a dame ju na standardnu velkost
            resize(croppedImage_original, noseposition_image, Size(200, 200), 0, 0, INTER_CUBIC);
            // vypocitame uhol rotacie pomocou histogramu orientovanych gradientov
            float rotation = gradienty(noseposition_image);
            if (rotation_def){
                rotate(srcImg, rotation, dstImg);
            }
            else {
                dstImg = srcImg;
            }
        }
        else{
            dstImg = srcImg;
        }
 
    }
    std::vector<Rect> faces;
    // Este raz detekujeme tvar na otocenom obrazku kvoli lepsej presnosti rozpoznavania tvare
    face_cascade.detectMultiScale(dstImg, faces, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING, Size(dstImg.size().width*0.2, dstImg.size().height*0.2));
 
    for (size_t i = 0; i < faces.size(); i++)
    {
 
        int tlY = faces[i].y;
        //tlY -= (faces[i].height / 3);
        if (tlY < 0){
            tlY = 0;
        }
        int drY = faces[i].y + faces[i].height;
        //drY += +(faces[i].height / 6);
        if (drY>dstImg.rows)
        {
            drY = dstImg.rows;
        }
        Point tl(faces[i].x, tlY);
        Point dr(faces[i].x + faces[i].width, drY);
 
        Rect myROI(tl, dr);
        // Orezeme na ksicht
        Mat croppedImage_original = dstImg(myROI);
        Mat croppedImageGray;
        // zmensime na standardnu velkost
        resize(croppedImage_original, crop, Size(width, height), 0, 0, INTER_CUBIC);
        //face_cascade.detectMultiScale(dstImg, faces, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING, Size(dstImg.size().width*0.2, dstImg.size().height*0.2));
        imshow("test", crop);
    }
 
    return crop;
}