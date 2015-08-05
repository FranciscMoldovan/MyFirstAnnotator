#include <uima/api.hpp>

#include <pcl/point_types.h>
#include <iai_rs/types/all_types.h>
//RS
#include <iai_rs/scene_cas.h>
#include <iai_rs/util/time.h>
#include <iai_rs/DrawingAnnotator.h>
//st
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ascii_io.h>
#include <pcl/point_types.h>

#include <iai_rs/util/output.h>
#include <iai_rs/util/wrapper.h>

#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

#include <pcl/impl/point_types.hpp>

//global descriptors:
//ESF
#include <pcl/features/esf.h>

//VFH
#include <pcl/features/vfh.h>

//CVFH
#include <pcl/features/cvfh.h>

//OUR-CVFH
#include <pcl/features/our_cvfh.h>

//GFPFH
#include <pcl/features/gfpfh.h>

//GRSD
#include <pcl/features/impl/grsd.h>

//local descriptors:
//PFH
#include<pcl/features/pfh.h>

//FPFH
#include<pcl/features/fpfh.h>

//RSD
#include<pcl/features/rsd.h>

//3DSC
#include<pcl/features/3dsc.h>

//USC
#include<pcl/features/usc.h>

//SHOT
#include<pcl/features/shot.h>

//SI
#include<pcl/features/spin_image.h>

//RIFT
#include<pcl/features/rift.h>
#include<pcl/point_types_conversion.h>
#include<pcl/features/intensity_gradient.h>

//NARF
#include<pcl/range_image/range_image.h>
#include<pcl/visualization/range_image_visualizer.h>


#define TESTINGM


typedef pcl::PointXYZRGBA PointT;
// A handy typedef.
typedef pcl::Histogram<135> ROPS135;

//Another handy typedef.
typedef pcl::Histogram<153> SpinImage;

//Another handy typedef.
typedef pcl::Histogram<32> RIFT32;
////////////////////////////////////
typedef std::vector<pcl::PointCloud<PointT>::Ptr> clusterVector;
////////////////////////

using namespace std;
using namespace cv;

using namespace uima;

static const Scalar colors[] = {
  Scalar(255,0,0),//{255,0,0},//RED
  Scalar(0,255,0),//{0,255,0},//GREEN
  Scalar(0,0,255),//{0,  0,255},//BLUE
  Scalar(255,165,0),//{255,165,0},//ORANGE
  Scalar(255,0,255),//{255,0,255},//MAGENTA
  Scalar(0,206,209),//{0,206,209},//DARK TURQOUISE
  Scalar(255,115,115)//{255,115,115}//PINKish
};

static const size_t nbOfColors = sizeof(colors)/sizeof(colors[0]);

class MyFirstAnnotator : public DrawingAnnotator
{
private:

  double pointSize;
  /**
   * @brief cloud_ptr-The point cloud extracted from CAS
   */
  pcl::PointCloud<PointT>::Ptr cloud_ptr;
  float test_param;
  std::vector<iai_rs::Cluster> clusters;
  int crtCluster=-1;
  //experimental
  clusterVector extractedClusters;
  /**
   * Arrays of Global Descriptors:
   */
  std::vector<pcl::ESFSignature640> descVectESF;
  std::vector<pcl::VFHSignature308> descVectVFH;
  std::vector<pcl::VFHSignature308> descVectCVFH;
  std::vector<pcl::VFHSignature308> descVectOUR_CVFH;
  std::vector<pcl::GFPFHSignature16> descVectGFPFH;
  std::vector<pcl::GRSDSignature21> descVectGRSD;
  //the much needed normals
  pcl::PointCloud<pcl::Normal>::Ptr normals;//(new pcl::PointCloud<pcl::Normal>);
  //the much needed KdTree structure
  pcl::search::KdTree<PointT>::Ptr kdtree;

  cv::Mat hists;
  cv::Mat color;

public:
  MyFirstAnnotator() : DrawingAnnotator(__func__), pointSize(1),
      cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGBA>),
      normals(new pcl::PointCloud<pcl::Normal>),
      kdtree(new pcl::search::KdTree<PointT>)
  {  }

  TyErrorId initialize(AnnotatorContext &ctx)
  {
    outInfo("initialize");
    ctx.extractValue("test_param", test_param);
    return UIMA_ERR_NONE;
  }

  TyErrorId typeSystemInit(TypeSystem const &type_system)
  {
    outInfo("typeSystemInit");
    return UIMA_ERR_NONE;
  }

  TyErrorId destroy()
  {
    outInfo("destroy");
    return UIMA_ERR_NONE;
  }

  /**
   * @brief saveToPCD-save pointcloud to *.pcd file
   * @param saveCloud-input point cloud
   */
  void saveToPCD(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr saveCloud)
  {
    pcl::io::savePCDFileASCII("my_cluster.pcd", *saveCloud);
  }

  /**
   * @brief drawHistograms -OpenCV histogram drawing
   * @param descVect -descriptor vector containing the data to draw histograms
   * @param title -the string to be displayed on the histogram (descriptor name)
   */
  template<typename globalDescSign>
  void drawHistograms(const std::vector<globalDescSign> &descVect, std::string title)
  {
    //each cluster's decriptor should have same nb of hist values
    int nbOfEntries=sizeof(descVect.at(0).histogram)/sizeof(float);
    int histWidthPixels=1650;
    int hist_h=480;
    cv::Mat histImage( hist_h+400, histWidthPixels, CV_8UC3, Scalar(173, 173, 173) );
    //Draw axes:
      //horizontal axis///////////////////////////////
      //body
      line(histImage, Point(9, hist_h+41),
                      Point(1630, hist_h+41),
                      Scalar(0,0,0), 1, 8, 0);
      //its arrow
        //upper part
        line(histImage, Point(1620, hist_h+36),
                        Point(1630, hist_h+41),
                        Scalar(0,0,0), 1, 8, 0);
        //lower part
        line(histImage, Point(1620, hist_h+46),
                        Point(1630, hist_h+41),
                        Scalar(0,0,0), 1, 8, 0);
    //////////////////////////////////////////////////

    //vertical axis///////////////////////////////////
      line(histImage, Point(9, hist_h+41),
                      Point(9, 20),
                      Scalar(0,0,0), 1, 8, 0);
      //its arrow
        //left part
        line(histImage, Point(4, 30),
                        Point(9, 20),
                        Scalar(0,0,0), 1, 8, 0);
        //right part
        line(histImage, Point(14, 30),
                        Point(9, 20),
                        Scalar(0,0,0), 1, 8, 0);
    ///////////////////////////////////////////////////

    /////////title of the histogram (descriptor name)/////////////////////////
    putText(histImage, title, cvPoint(30,30),
        FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(102,0,102), 1, CV_AA);
    //////////////////////////////////////////////////////////////////////////

    for(int i = 0; i < clusters.size(); i++)
    {
      //the key to the diagram////////////////////////////////////////////////////
      cv::rectangle(
        histImage,
        cv::Point(10, hist_h+60+20*i),
        cv::Point(30, hist_h+60+20*i+10),
        colors[i%nbOfColors],
        CV_FILLED
      );
      putText(histImage, "Cluster#"+std::to_string(i), cvPoint(30, hist_h+70+20*i),
        FONT_HERSHEY_COMPLEX_SMALL, 0.8, colors[i%nbOfColors], 1, CV_AA);
      ////////////////////////////////////////////////////////////////////////////
      float maxHist=0;
        for(int j = 1; j < nbOfEntries; j++)
        {
          if(descVect.at(i).histogram[j]>maxHist)
            maxHist=descVect.at(i).histogram[j];
        }
       for (int j = 1; j < nbOfEntries; j++)
       {
         float heightOfEntry;
         float pastEntryHeight;
         heightOfEntry=hist_h*(descVect.at(i).histogram[j])/maxHist;
         pastEntryHeight=hist_h*(descVect.at(i).histogram[j-1]/maxHist);
         line(histImage, Point(10+((histWidthPixels)/nbOfEntries)*j, hist_h-heightOfEntry+40),
                         Point(10+((histWidthPixels)/nbOfEntries)*(j-1), hist_h-pastEntryHeight+40),
                         colors[i % nbOfColors], 1, 8, 0);
       }
     }
     hists = histImage.clone();
  }


  /**
 * @brief extractClusters-func to extract clusters from input cloud
 * @param inputCloud-a single point cloud
 * @return extractedClusters-vector of clusters extracted
 */
clusterVector extractClusters()
{   
  //first, empty the vector of clusters
  extractedClusters.clear();
  //next, iterate over clusters
  for(int i = 0; i < clusters.size(); ++i)
  {
    iai_rs::Cluster &cluster = clusters[i];
    if(!cluster.points.has())
    {
      continue;
    }
    pcl::PointIndicesPtr indices(new pcl::PointIndices());
    iai_rs::conversion::from(((iai_rs::ReferenceClusterPoints)cluster.points.get()).indices.get(), *indices);
    pcl::PointCloud<PointT>::Ptr cluster_cloud(new pcl::PointCloud<PointT>());
    pcl::ExtractIndices<PointT> ei;
    ei.setInputCloud(cloud_ptr);
    ei.setIndices(indices);
    ei.filter(*cluster_cloud);
    extractedClusters.push_back(cluster_cloud);
  }
  return extractedClusters;
}

/**
 * @brief calcESF
 * Function that calculates ESF (Ensemble of Shape Functions)
 * ~~Global Descriptor~~
 */
void calcESF()
{
  extractedClusters=extractClusters();
  for (int i = 0; i < extractedClusters.size(); ++i)
  {
    //Object for storing the ESF descriptor
    pcl::PointCloud<pcl::ESFSignature640>::Ptr descriptor(new pcl::PointCloud<pcl::ESFSignature640>);
    //ESF estimation object
    pcl::ESFEstimation<pcl::PointXYZRGBA, pcl::ESFSignature640>esf;
    esf.setInputCloud(extractedClusters.at(i));
    esf.compute(*descriptor);
      descVectESF.push_back(descriptor->points[0]);
  }
}

/**
 * @brief computeNormals -computes normals for the input point cloud
 * @param inputCloud
 * @return normals -the class field that stores the normals
 */
pcl::PointCloud<pcl::Normal>::Ptr computeNormals(pcl::PointCloud<PointT>::Ptr &inputCloud)
{
    // Estimate the normals.
    pcl::NormalEstimation<PointT, pcl::Normal> normalEstimation;
    normalEstimation.setInputCloud(inputCloud);
    normalEstimation.setRadiusSearch(0.03);
    normalEstimation.setSearchMethod(kdtree);
    normalEstimation.compute(*normals);
    return normals;
}

/**
 * @brief calcVFH
 *Function that calculates VFH (Viewpoint Feature Histogram)
 *~~Global Descriptor~~
 */
void calcVFH()
{
  extractedClusters=extractClusters();
  //Object for storing the VFH descriptor
  pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptor(new pcl::PointCloud<pcl::VFHSignature308>);
  for(int i = 0; i < extractedClusters.size(); ++i)
  {
    //compute normals
    normals=computeNormals(extractedClusters.at(i));
    // VFH estimation object.
    pcl::VFHEstimation<PointT, pcl::Normal, pcl::VFHSignature308> vfh;
    vfh.setInputCloud(extractedClusters.at(i));
    vfh.setInputNormals(normals);
    vfh.setSearchMethod(kdtree);
    // Optionally, we can normalize the bins of the resulting histogram,
    // using the total number of points.
    vfh.setNormalizeBins(true);
    // Also, we can normalize the SDC with the maximum size found between
    // the centroid and any of the cluster's points.
    vfh.setNormalizeDistance(false);
    vfh.compute(*descriptor);
      descVectVFH.push_back(descriptor->points[0]);
  }
}

/**
 * @brief calcCVFH
 *Function that calculates CVFH (Clustered Viewpoint Feature Histogram)
 *~~Global Descriptor~~
 */
void calcCVFH()
{
  extractedClusters=extractClusters();
  //Object for storing the VFH descriptor
  pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptor(new pcl::PointCloud<pcl::VFHSignature308>);
  for(int i = 0; i < extractedClusters.size(); ++i)
  {
    //compute normals
    normals=computeNormals(extractedClusters.at(i));
    //CVFH estimation object.
    pcl::CVFHEstimation<PointT, pcl::Normal, pcl::VFHSignature308> cvfh;
    cvfh.setInputCloud(extractedClusters.at(i));
    cvfh.setInputNormals(normals);
    cvfh.setSearchMethod(kdtree);
    //set maximum allowable derivation of the normals,
    //for the region segmentation step.
    cvfh.setEPSAngleThreshold(5.0/180.0*M_PI);//5 deg
    //Set the curvature threshol (maximum disparity between curvatures),
    //for the region segmentation step.
    cvfh.setCurvatureThreshold(1.0);
    //Set to true to normalize the bins of the resulting hist,
    //using the total number of points. Note:enabling it will make
    //CVFH invariant to scale, just like VFH, but the authors encourage
    //the opposite.
    cvfh.setNormalizeBins(false);
    cvfh.compute(*descriptor);
      descVectCVFH.push_back(descriptor->points[0]);
  }
}



//******//DOES NOT WORK
//******/**
//****** * @brief calcOUR_CVFH
//****** * Function that calculates OUR-CVFH (Oriented, Unique and Repeatable CVFH)
//****** * ~~Global Descriptor~~
//****** */
//******void calcOUR_CVFH()
//******{
//******  extractedClusters=extractClusters();
//******  for (int i = 0; i < extractedClusters.size(); ++i)
//******  {
//******    //Object for storing the OUR-CVFH descriptor.
//******    pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptors(new pcl::PointCloud<pcl::VFHSignature308>);
//******    //Note: you should have performed preprocessing to cluster out the object
//******    //from the cloud, and save to an individual file
//******    //compute normals:
//******    normals=computeNormals(extractedClusters.at(i));
//******    //OUR-CVFH estimation object.
//******    pcl::OURCVFHEstimation<PointT, pcl::Normal, pcl::VFHSignature308>ourcvfh;
//******    ourcvfh.setInputCloud(extractedClusters.at(i));
//******    ourcvfh.setInputNormals(normals);
//******    ourcvfh.setSearchMethod(kdtree);
//******    ourcvfh.setEPSAngleThreshold(5.0/180*M_PI);//5 deg
//******    ourcvfh.setCurvatureThreshold(1.0);
//******    ourcvfh.setNormalizeBins(false);
//******    //Set the minimum axis ratio between the SGURF axes. At the disambiguation phase,
//******    //this will decide if additional Reference Frames need to be created, id ambiguous.
//******    ourcvfh.setAxisRatio(0.8);
//******    ourcvfh.compute(*descriptors);
//******
//******      descVectOUR_CVFH.push_back(descriptors->points[0]);
//******  }
//******}


/**
 * @brief calcGFPFH
 * Function that calculates GFPFH (Global Fast Point Feature Histogram)
 * ~~Global Descriptor~~
 */
void calcGFPFH()
{
  extractedClusters=extractClusters();
  for(int i = 0; i < extractedClusters.size(); ++i)
  {
    //Object for storing the GFPFH descriptor.
    pcl::PointCloud<pcl::GFPFHSignature16>::Ptr descriptor(new pcl::PointCloud<pcl::GFPFHSignature16>);
    //Note: you should have performed preprocessing to cluster out the object
    //from the cloud, and save it to an individual file.

    //save each cluster to PCD, so it can be converted to XYZL
    saveToPCD(extractedClusters.at(i));
    // Cloud for storing the object.
    pcl::PointCloud<pcl::PointXYZL>::Ptr object(new pcl::PointCloud<pcl::PointXYZL>);
    // Note: you should now perform classification on the cloud's points. See the
    // original paper for more details. For this example, we will now consider 4
    // different classes, and randomly label each point as one of them.

    // Read a PCD file from disk.
    if(pcl::io::loadPCDFile<pcl::PointXYZL>("my_cluster.pcd", *object) != 0)
    {
        outInfo("READ ERROR!!!!");
    } else
        outInfo("OK READ OK!!!!");

    for(int var = 0; var < object->points.size(); ++var)
    {
       object->points[i].label = 1+i%4;
    }
    //ESF estimation object;
    pcl::GFPFHEstimation<pcl::PointXYZL,pcl::PointXYZL,pcl::GFPFHSignature16>gfpfh;
    gfpfh.setInputCloud(object);
    //Set the object that contains the labels for each point. Thanks to the
    //PointXYZL type, we can use the same object we store the cloud in.
    gfpfh.setInputLabels(object);
    //Set the size of the octree leaves to 1cm(cubic)
    gfpfh.setOctreeLeafSize(0.01);
    //Set the number of classes the cloud has been labeled with
    //(default is 16)
    gfpfh.setNumberOfClasses(4);
    gfpfh.compute(*descriptor);
      descVectGFPFH.push_back(descriptor->points[0]);
  }
}

//DOES NOT EXIST IN MY VERSION OF PCL
//******* void calcGRSD()
//******* {
//*******     //iterate over clusters
//*******     for(size_t i = 0; i < clusters.size(); ++i)
//*******     {
//*******        iai_rs::Cluster &cluster = clusters[i];
//*******        if(!cluster.points.has())
//*******        {
//*******          continue;
//*******        }
//*******        pcl::PointIndicesPtr indices(new pcl::PointIndices());
//*******        iai_rs::conversion::from(((iai_rs::ReferenceClusterPoints)cluster.points.get()).indices.get(), *indices);
//*******        pcl::PointCloud<PointT>::Ptr cluster_cloud(new pcl::PointCloud<PointT>());
//*******        pcl::ExtractIndices<PointT> ei;
//*******        ei.setInputCloud(cloud_ptr);
//*******        ei.setIndices(indices);
//*******        ei.filter(*cluster_cloud);
//*******
//*******        //Object for storing the normals.
//*******        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
//*******        //Object for storing the GRSD descriptors for each point.
//*******        pcl::PointCloud<pcl::GRSDSignature21>::Ptr descriptors(new pcl::PointCloud<pcl::GRSDSignature21>());
//*******
//*******        //Estimate the normals.
//*******        pcl::NormalEstimation<PointT,pcl::Normal>normalEstimation;
//*******        normalEstimation.setInputCloud(cluster_cloud);
//*******        normalEstimation.setRadiusSearch(0.03);
//*******        pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
//*******        normalEstimation.setSearchMethod(kdtree);
//*******        normalEstimation.compute(*normals);
//*******
//*******        //GRSD estimation object.
//*******        //GRSDSignature21
//*******        // GRSD estimation object.
//*******            //GRSDEstimation<PointXYZ, Normal, GRSDSignature21> grsd;
//      }
//  }

/**
 * @brief calcPFH
 * Function that calculates PFH(Point Feature Histogram)
 * ~~Local Descriptor~~
 */
void calcPFH()
{
    extractedClusters=extractClusters();
    for(int i = 0; i < extractedClusters.size(); ++i)
    {
    //Object for storing the normals.
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    //Object for storing the PFH descriptors for each point.
    pcl::PointCloud<pcl::PFHSignature125>::Ptr descriptors(new pcl::PointCloud<pcl::PFHSignature125>());
    // Note: you would usually perform downsampling now. It has been omitted here
    // for simplicity, but be aware that computation can take a long time.
    normals=computeNormals(extractedClusters.at(i));
    //PFH estimation object.
    pcl::PFHEstimation<PointT,pcl::Normal,pcl::PFHSignature125>pfh;
    pfh.setInputCloud(extractedClusters.at(i));
    pfh.setInputNormals(normals);
    pfh.setSearchMethod(kdtree);
    //Search radius, to look for neighbours. Note: the value given here has
    //to be larger than the radius used to estimate the normals.
    pfh.setRadiusSearch(0.05);
      pfh.compute(*descriptors);
  }
}


//  void calcFPFH()
//  {
//      //iterate over clusters
//      for(size_t i = 0; i < clusters.size(); ++i)
//      {
//         iai_rs::Cluster &cluster = clusters[i];
//         if(!cluster.points.has())
//         {
//           continue;
//         }
//         pcl::PointIndicesPtr indices(new pcl::PointIndices());
//         iai_rs::conversion::from(((iai_rs::ReferenceClusterPoints)cluster.points.get()).indices.get(), *indices);
//         pcl::PointCloud<PointT>::Ptr cluster_cloud(new pcl::PointCloud<PointT>());
//         pcl::ExtractIndices<PointT> ei;
//         ei.setInputCloud(cloud_ptr);
//         ei.setIndices(indices);
//         ei.filter(*cluster_cloud);


//         ///testing:
//  #ifdef TESTING

//         pcl::PointCloud<PointT>::Ptr cluster_cloud_piece(new pcl::PointCloud<PointT>());

//         int nrTestPoints=0;

//         if (cluster_cloud->size()<100)
//            nrTestPoints=cluster_cloud->size();
//         else
//            nrTestPoints=100;

//         for (int var = 0; var < nrTestPoints; ++var)
//         {
//            cluster_cloud_piece->points.push_back(cluster_cloud->points.at(i));
//         }

//         outInfo("asdasdas");

//         /// ......///////

//         //Object for storing the normals.
//         pcl::PointCloud<pcl::Normal>::Ptr normals_piece(new pcl::PointCloud<pcl::Normal>);
//         //Object for storing the FPFH descriptors for each point.
//         pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors(new pcl::PointCloud<pcl::FPFHSignature33>());

//         //Estimate the normals.
//         pcl::NormalEstimation<PointT, pcl::Normal>normalEstimationTest;
//         normalEstimationTest.setInputCloud(cluster_cloud_piece);
//         normalEstimationTest.setRadiusSearch(0.03);
//         pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
//         normalEstimationTest.setSearchMethod(kdtree);
//         normalEstimationTest.compute(*normals_piece);

//         //FPFH estimation object.
//         pcl::FPFHEstimation<PointT,pcl::Normal,pcl::FPFHSignature33>fpfh;
//         fpfh.setInputCloud(cluster_cloud_piece);
//         fpfh.setInputNormals(normals_piece);
//         fpfh.setSearchMethod(kdtree);
//         //Search radius, to look for neighbours. Note: the value given here has to be
//         //larger thatn the radius used to estimate the normals.
//         fpfh.setRadiusSearch(0.05);
//         fpfh.compute(*descriptors);

//#endif

//#ifndef TESTING
//         //Object for storing the normals.
//         pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
//         //Object for storing the FPFH descriptors for each point.
//         pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors(new pcl::PointCloud<pcl::FPFHSignature33>());

//         // Note: you would usually perform downsampling now. It has been omitted here
//         // for simplicity, but be aware that computation can take a long time.

//         //Estimate the normals.
//         pcl::NormalEstimation<PointT, pcl::Normal>normalEstimation;
//         normalEstimation.setInputCloud(cluster_cloud);
//         normalEstimation.setRadiusSearch(0.03);
//         pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
//         normalEstimation.setSearchMethod(kdtree);
//         normalEstimation.compute(*normals);

//         //FPFH estimation object.
//         pcl::FPFHEstimation<PointT,pcl::Normal,pcl::FPFHSignature33>fpfh;
//         fpfh.setInputCloud(cluster_cloud);
//         fpfh.setInputNormals(normals);
//         fpfh.setSearchMethod(kdtree);
//         //Search radius, to look for neighbours. Note: the value given here has to be
//         //larger thatn the radius used to estimate the normals.
//         fpfh.setRadiusSearch(0.05);
//         fpfh.compute(*descriptors);
//         outInfo("asdasdas");
//#endif


//      }
//  }


//  //NO RSD, SORRY
//  /**
//   * @brief calcRSD
//   * NOT INCLUDED IN CURRENT PCL VERSION
//   */
//  void calcRSD()
//  {
//      //iterate over clusters
//      for(size_t i = 0; i < clusters.size(); ++i)
//      {
//         iai_rs::Cluster &cluster = clusters[i];
//         if(!cluster.points.has())
//         {
//           continue;
//         }
//         pcl::PointIndicesPtr indices(new pcl::PointIndices());
//         iai_rs::conversion::from(((iai_rs::ReferenceClusterPoints)cluster.points.get()).indices.get(), *indices);
//         pcl::PointCloud<PointT>::Ptr cluster_cloud(new pcl::PointCloud<PointT>());
//         pcl::ExtractIndices<PointT> ei;
//         ei.setInputCloud(cloud_ptr);
//         ei.setIndices(indices);
//         ei.filter(*cluster_cloud);


//         //Object for storing the normals.
//         pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
//         //Object for storing the RSD

//         // Note: you would usually perform downsampling now. It has been omitted here
//         // for simplicity, but be aware that computation can take a long time.

//         //Estimate the normals.
//         pcl::NormalEstimation<PointT,pcl::Normal>normalEstimation;
//         normalEstimation.setInputCloud(cluster_cloud);
//         normalEstimation.setRadiusSearch(0.03);
//         pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
//         normalEstimation.setSearchMethod(kdtree);
//         normalEstimation.compute(*normals);

//         //RSD estimation object.
//         //RSDEstimation<PointT,pcl::Normal,pcl::PrincipalRadiiRSD>rsd;
//         //RSDE

//      }
//  }


//  void calc3DSC()
//  {
//      //iterate over clusters
//      for(size_t i = 0; i < clusters.size(); ++i)
//      {
//         iai_rs::Cluster &cluster = clusters[i];
//         if(!cluster.points.has())
//         {
//           continue;
//         }
//         pcl::PointIndicesPtr indices(new pcl::PointIndices());
//         iai_rs::conversion::from(((iai_rs::ReferenceClusterPoints)cluster.points.get()).indices.get(), *indices);
//         pcl::PointCloud<PointT>::Ptr cluster_cloud(new pcl::PointCloud<PointT>());
//         pcl::ExtractIndices<PointT> ei;
//         ei.setInputCloud(cloud_ptr);
//         ei.setIndices(indices);
//         ei.filter(*cluster_cloud);

//         //Object for storing the normals.
//         pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
//         //Object for storing the 3DSC descriptors for each point.
//         pcl::PointCloud<pcl::ShapeContext1980>::Ptr descriptors(new pcl::PointCloud<pcl::ShapeContext1980>());

//         // Note: you would usually perform downsampling now. It has been omitted here
//         // for simplicity, but be aware that computation can take a long time.

//         //Estimate the normals.
//         pcl::NormalEstimation<PointT,pcl::Normal>normalEstimation;
//         normalEstimation.setInputCloud(cluster_cloud);
//         normalEstimation.setRadiusSearch(0.03);
//         pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
//         normalEstimation.setSearchMethod(kdtree);
//         normalEstimation.compute(*normals);


//         //3DSC estmation object.
//         pcl::ShapeContext3DEstimation<PointT,pcl::Normal,pcl::ShapeContext1980>sc3d;
//         sc3d.setInputCloud(cluster_cloud);
//         sc3d.setInputNormals(normals);
//         sc3d.setSearchMethod(kdtree);
//         //Search radius, to look for neighbours. It will also be the radius of the
//         //support shphere
//         sc3d.setRadiusSearch(0.05);
//         //The minimal radius value for each sphere, to avoid being too sensitive
//         //in bins close to the sphere center.
//         sc3d.setMinimalRadius(0.05/10.0);
//         //Radius used to compute the local point density for the neighbours
//         //(the density is the number of points within that radius).
//         sc3d.setPointDensityRadius(0.05/5.0);
//         sc3d.compute(*descriptors);
//      }
//  }


//  /**
//   * @brief calcUSC
//   * NOT WORKIIIING
//   */
//  void calcUSC()
//  {
//      //iterate over clusters
//      for(size_t i = 0; i < clusters.size(); ++i)
//      {
//         iai_rs::Cluster &cluster = clusters[i];
//         if(!cluster.points.has())
//         {
//           continue;
//         }
//         pcl::PointIndicesPtr indices(new pcl::PointIndices());
//         iai_rs::conversion::from(((iai_rs::ReferenceClusterPoints)cluster.points.get()).indices.get(), *indices);
//         pcl::PointCloud<PointT>::Ptr cluster_cloud(new pcl::PointCloud<PointT>());
//         pcl::ExtractIndices<PointT> ei;
//         ei.setInputCloud(cloud_ptr);
//         ei.setIndices(indices);
//         ei.filter(*cluster_cloud);

//         //Object for storing the USC descriptors for each point.
//         pcl::PointCloud<pcl::UniqueShapeContext1960>::Ptr descriptors(new pcl::PointCloud<pcl::UniqueShapeContext1960>());

//         // Note: you would usually perform downsampling now. It has been omitted here
//         // for simplicity, but be aware that computation can take a long time.

//         //USC estimation object.

//         //dice nu merrrrrrri?
//         pcl::UniqueShapeContext<pcl::PointXYZ, pcl::UniqueShapeContext1960, pcl::ReferenceFrame> usc;

//         //   usc.setInputCloud(cluster_cloud);
//         //Search radius, to look for neighbours. It will also be the radius of the
//         //support sphere.
// //        usc.setRadiusSearch(0.05);
//         //The minimal radius value for the search sphere, to avoid being too
//         //sensitive in bins close to the center of the sphere.
// //        usc.setRadiusSearch(0.05/10.0);
//         //Radius used to compute the local point density for the neighbours
//         //(the density is the number of points within that radius).
//   //      usc.setPointDensityRadius(0.05/5.0);
//         //Set the radius to compute the local Reference Frame.
//   //      usc.setLocalRadius(0.05);

//    //     usc.compute(*descriptors);
//      }
//  }


//  void calcSHOT()
//  {
//      //iterate over clusters
//      for(size_t i = 0; i < clusters.size(); ++i)
//      {
//         iai_rs::Cluster &cluster = clusters[i];
//         if(!cluster.points.has())
//         {
//           continue;
//         }
//         pcl::PointIndicesPtr indices(new pcl::PointIndices());
//         iai_rs::conversion::from(((iai_rs::ReferenceClusterPoints)cluster.points.get()).indices.get(), *indices);
//         pcl::PointCloud<PointT>::Ptr cluster_cloud(new pcl::PointCloud<PointT>());
//         pcl::ExtractIndices<PointT> ei;
//         ei.setInputCloud(cloud_ptr);
//         ei.setIndices(indices);
//         ei.filter(*cluster_cloud);

//         //Object for storing the normals.
//         pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
//         //Object for storing the SHOT descriptors for each point.
//         pcl::PointCloud<pcl::SHOT352>::Ptr descriptors(new pcl::PointCloud<pcl::SHOT352>());

//         // Note: you would usually perform downsampling now. It has been omitted here
//         // for simplicity, but be aware that computation can take a long time.

//         //Estimate the normals.
//         pcl::NormalEstimation<PointT,pcl::Normal>normalEstimation;
//         normalEstimation.setInputCloud(cluster_cloud);
//         normalEstimation.setRadiusSearch(0.03);
//         pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
//         normalEstimation.setSearchMethod(kdtree);
//         normalEstimation.compute(*normals);

//         //SHOT estimation object.
//         pcl::SHOTEstimation<PointT, pcl::Normal, pcl::SHOT352>shot;
//         shot.setInputCloud(cluster_cloud);
//         shot.setInputNormals(normals);
//         //The radius that defines which of the keypoint's neighbours are described.
//         //If too large, there may be clutter, and if too small, not enough points may be found.
//         shot.setRadiusSearch(0.02);
//         shot.compute(*descriptors);
//      }
//  }


//  void calcSI()
//  {
//      //iterate over clusters
//      for(size_t i = 0; i < clusters.size(); ++i)
//      {
//         iai_rs::Cluster &cluster = clusters[i];
//         if(!cluster.points.has())
//         {
//           continue;
//         }
//         pcl::PointIndicesPtr indices(new pcl::PointIndices());
//         iai_rs::conversion::from(((iai_rs::ReferenceClusterPoints)cluster.points.get()).indices.get(), *indices);
//         pcl::PointCloud<PointT>::Ptr cluster_cloud(new pcl::PointCloud<PointT>());
//         pcl::ExtractIndices<PointT> ei;
//         ei.setInputCloud(cloud_ptr);
//         ei.setIndices(indices);
//         ei.filter(*cluster_cloud);

//         //Object for storing the normals.
//         pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
//         //Object for storing the Spin Image for each point.
//         pcl::PointCloud<SpinImage>::Ptr descriptors(new pcl::PointCloud<SpinImage>());


//         // Note: you would usually perform downsampling now. It has been omitted here
//         // for simplicity, but be aware that computation can take a long time.

//         //Estimate the normals.
//         pcl::NormalEstimation<PointT,pcl::Normal>normalEstimation;
//         normalEstimation.setInputCloud(cluster_cloud);
//         normalEstimation.setRadiusSearch(0.03);
//         pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
//         normalEstimation.setSearchMethod(kdtree);
//         normalEstimation.compute(*normals);

//         //Spin Image estimation object.
//         pcl::SpinImageEstimation<PointT,pcl::Normal,SpinImage>si;
//         si.setInputCloud(cluster_cloud);
//         si.setInputNormals(normals);
//         //Radius of the support cylinder.
//         si.setRadiusSearch(0.02);
//         //Set the resolution of the spin image
//         //(the number of bins along one dimension).
//         //Note:you must change the output histogram size to reflect this.
//         si.setImageWidth(8);
//         si.compute(*descriptors);

//      }
//  }


//  void calcRIFT()
//  {
//      //iterate over clusters
//      for(size_t i = 0; i < clusters.size(); ++i)
//      {
//         iai_rs::Cluster &cluster = clusters[i];
//         if(!cluster.points.has())
//         {
//           continue;
//         }
//         pcl::PointIndicesPtr indices(new pcl::PointIndices());
//         iai_rs::conversion::from(((iai_rs::ReferenceClusterPoints)cluster.points.get()).indices.get(), *indices);
//         pcl::PointCloud<PointT>::Ptr cluster_cloud(new pcl::PointCloud<PointT>());
//         pcl::ExtractIndices<PointT> ei;
//         ei.setInputCloud(cloud_ptr);
//         ei.setIndices(indices);
//         ei.filter(*cluster_cloud);

//        //Object for storing the point cloud with color information.
//        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudColor(new pcl::PointCloud<pcl::PointXYZRGB>);
//        //Object for storing the point cloud with intensity value.
//        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudIntensity(new pcl::PointCloud<pcl::PointXYZI>);
//        //Object for storing the intensity gradients.
//        pcl::PointCloud<pcl::IntensityGradient>::Ptr gradients(new pcl::PointCloud<pcl::IntensityGradient>);
//        //Object for storing the normals.
//        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
//        //Object for storing the RIFT descriptor for each point.
//        pcl::PointCloud<RIFT32>::Ptr descriptors(new pcl::PointCloud<RIFT32>());

//        // Note: you would usually perform downsampling now. It has been omitted here
//        // for simplicity, but be aware that computation can take a long time.

//        //need to save to PCD, and retrieve RGBXYZ ! (for data type consistency)
//        saveToPCD(cluster_cloud);

//        // Read a PCD file from disk.
//        if (pcl::io::loadPCDFile<pcl::PointXYZRGB>("my_cluster.pcd", *cloudColor) != 0)
//        {
//            outInfo("ERROR!");
//        }


//        //Convert the RGB to intensity.
//        pcl::PointCloudXYZRGBtoXYZI(*cloudColor,*cloudIntensity);

//        //Estimate the normals.
//        pcl::NormalEstimation<pcl::PointXYZI,pcl::Normal>normalEstimation;
//        normalEstimation.setInputCloud(cloudIntensity);
//        normalEstimation.setRadiusSearch(0.03);
//        pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZI>);
//        normalEstimation.setSearchMethod(kdtree);
//        normalEstimation.compute(*normals);

//        //compute the intensity gradients.
//        pcl::IntensityGradientEstimation<pcl::PointXYZI,pcl::Normal,pcl::IntensityGradient,
//                pcl::common::IntensityFieldAccessor<pcl::PointXYZI> >ge;
//        ge.setInputCloud(cloudIntensity);
//        ge.setInputNormals(normals);
//        ge.setRadiusSearch(0.03);
//        ge.compute(*gradients);

//        //RIFT estimation object.
//        pcl::RIFTEstimation<pcl::PointXYZI,pcl::IntensityGradient,RIFT32>rift;
//        rift.setInputCloud(cloudIntensity);
//        rift.setSearchMethod(kdtree);
//        //Set the intensity gradients to use.
//        rift.setInputGradient(gradients);
//        //Radius, to get all the neighbours within.
//        rift.setRadiusSearch(0.02);
//        //Set the number of bins to use in the distance dimension.
//        rift.setNrDistanceBins(4);
//        //Set the number of bins to use in the gradient orientation dimension.
//        rift.setNrGradientBins(8);
//        //Note:you must change the output histogram size to reflect the previous values.

//        rift.compute(*descriptors);

//      }
//  }



//  void calcNARF()
//  {
//      //iterate over clusters
//      for(size_t i = 0; i < clusters.size(); ++i)
//      {
//         iai_rs::Cluster &cluster = clusters[i];
//         if(!cluster.points.has())
//         {
//           continue;
//         }
//         pcl::PointIndicesPtr indices(new pcl::PointIndices());
//         iai_rs::conversion::from(((iai_rs::ReferenceClusterPoints)cluster.points.get()).indices.get(), *indices);
//         pcl::PointCloud<PointT>::Ptr cluster_cloud(new pcl::PointCloud<PointT>());
//         pcl::ExtractIndices<PointT> ei;
//         ei.setInputCloud(cloud_ptr);
//         ei.setIndices(indices);
//         ei.filter(*cluster_cloud);

//         //Parameters needed by the range image object:

//         //Angular resolution if the angular distance between pixels.
//         //Kinect:57 deg horiz FOV, 43 deg vert FOV 640x480(chosen here)
//         //Xtion:58 deg horiz FOV, 45 deg vert FOV, 640x480.
//         float angularResolutionX=(float)(57.0f/640.0f*(M_PI/180.0f));
//         float angularResolutionY=(float)(43.0f/480.0f*(M_PI/180.0f));
//         //Maximum horizontal and vertical angles. For example, for a null
//         //panoramic scan, the first would be 360 deg. Chooseing values
//         //that adjust to the real sensor will decrease the time it takes,
//         //but don't worry. If the values are bigger than the real ones,
//         //the image will be automatically cropped to discard empty zones.
//         float maxAngleX=(float)(60.0f*(M_PI/180.0f));
//         float maxAngleY=(float)(50.0f*(M_PI/180.0f));
//         //Sensor pose. Thankfully, the cloud includes the data.
//         Eigen::Affine3f sensorPose=Eigen::Affine3f(Eigen::Translation3f
//                                    (
//                                    cluster_cloud->sensor_origin_[0],
//                                    cluster_cloud->sensor_origin_[1],
//                                    cluster_cloud->sensor_origin_[2]
//                                    ))   *
//                 Eigen::Affine3f(cluster_cloud->sensor_orientation_);
//         //Noise level. If greater than 0, values of neighbouring points
//         //will be averaged. This would set the search radius(e.g., 0.03==3cm).
//         float noiseLevel=0.0f;
//         //Minimum range. If set, any point closer to the sensor than this will
//         //be ignored.
//         float minimumRange=0.0f;
//         //Border size. If greater than 0, a border of "unobserved" points will be
//         //left in the image when it is cropped.
//         int borderSize=1;

//         //Range image object.
//         pcl::RangeImage rangeImage;
//         rangeImage.createFromPointCloud(*cluster_cloud,
//                    angularResolutionX,angularResolutionY,
//                    maxAngleX, maxAngleY, sensorPose, pcl::RangeImage::CAMERA_FRAME,
//                    noiseLevel, minimumRange, borderSize);

//         //Visualize the image.
//         pcl::visualization::RangeImageVisualizer viewer("Range image");
//         viewer.showRangeImage(rangeImage);
//         while(!viewer.wasStopped())
//         {
//             viewer.spinOnce();
//             //Sleep 100ms to go easy on the CPU.
//             pcl_sleep(0.1);
//         }
//      }
//  }

  TyErrorId processWithLock(CAS &tcas, ResultSpecification const &res_spec)
  {
    outInfo("process start");
    iai_rs::util::StopWatch clock;
    iai_rs::SceneCas cas(tcas);
    clusters.clear();
    //retrieve the point cloud to be processed:
    cas.getPointCloud(*cloud_ptr);
    iai_rs::SceneWrapper scene(cas.getScene());
    //2.filter out clusters into array
    scene.identifiables.filter(clusters);
    outInfo("Number of clusters:" << clusters.size());

    /************************
    *                       *
    *  GLOBAL DESCRIPTORS   *
    *                       *
    * ***********************/
      //TESTED-OK
    ///1.ESF
    //descVectESF.clear();
    //calcESF();
    //drawHistograms(descVectESF, "Ensemble of Shape Functions (ESF)");
    ///////

      //TESTED-OK
    ///2.VFH
    //descVectVFH.clear();
    //calcVFH();
    //drawHistograms(descVectVFH, "Viewpoint Feature Histogram (VFH)");
    ////////

      //TESTED-OK
    ///3.CVFH
    //descVectCVFH.clear();
    //calcCVFH();
    //drawHistograms(descVectCVFH, "Clustered Viewpoint Feature Histogram (CVFH)");
    ////////

      //TESTED-NOT OK
    ///4.OUR-CVFH
    //descVectOUR_CVFH.clear();
    //calcOUR_CVFH();
    //drawHistograms(descVectOUR_CVFH, "Clustered Viewpoint Feature Histogram (CVFH)");
    ////////

      //TESTED-OK
      //highly inefficient implementation, haha
    ///5.GFPFH
    //descVectGFPFH.clear();
    //calcGFPFH();
    //drawHistograms(descVectGFPFH, "Global Fast Point Feature Histogram (GFPFH)");
    ////////

      //TESTED-NOT YET..
    //6.GRSD - Can't try it yet.

   /************************
   *                       *
   *   LOCAL DESCRIPTORS   *
   *                       *
   * ***********************/

      //TESTED-OK
    //1.PFH
    //calcPFH();
    /////////////

    //2.FPFH
    //calcFPFH();

    //3.RSD
    //calcRSD();

    //4.3DSC
    //calc3DSC();
    ///////////

    //5.USC
    //calcUSC();
    /////////

    //6.SHOT
    //calcSHOT();
    ////////

    //7.SI
    //calcSI();
    ////////

    //works
    //8.RIFT
    //calcRIFT();
    //////////

    // I have some idea why it does not work :(
    // I don't know how to fix it
    //9.NARF
    //calcNARF();
    //////////

    //////////your code goes here!
    //saveToPCD(cloud_ptr_clean);
    //////////////////////////////
     outInfo("took: " << clock.getTime() << " ms.");


    return UIMA_ERR_NONE;
  }
  void drawImageWithLock(cv::Mat &disp)
  {
     disp=hists.clone();
  }

  void fillVisualizerWithLock(pcl::visualization::PCLVisualizer &visualizer, const bool firstRun)
  {
    //const std::string &cloudname = this->name;

    if(firstRun)
    {
      visualizer.addPointCloud(cloud_ptr, "cloudname");
      visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointSize, "cloudname");
    }
    else
    {
     visualizer.updatePointCloud(cloud_ptr, "cloudname");
     visualizer.getPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointSize, "cloudname");
     visualizer.removeAllShapes();
    }
  }

};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(MyFirstAnnotator)



























