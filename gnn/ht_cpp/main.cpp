#include <iostream>
#include <vector>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class HoughHit {
  public:

    HoughHit() = default;

    float x;
    float y;
    float t;
    float x2plusy2;
};

class HoughTransform {
  public:
  
  HoughTransform(std::vector<float> x, std::vector<float> y, std::vector<float> t){
    std::cout << "HoughTransform constructor" << std::endl;
    ht(x, y, t);
  }

  std::vector<std::vector<float>> result;
  std::vector<std::vector<int>> indices;

  private:

  float fTimeCut = 3.0; //3.0
  
  float fMinDistance = 2.5; //2.5
  float fMinDistanceSq = fMinDistance * fMinDistance;
  float fMaxDistance = 11.5;//11.5
  float fMaxDistanceSq = fMaxDistance * fMaxDistance;
  
  float fMinRadius = 3.5;
  float fMaxRadius = 5.7;
  
  int fNofBinsX = 25;
  int fNofBinsY = 25;
  int fNofBinsR = 32;

  float fDx        = 2.f * fMaxDistance / (float) fNofBinsX;
  float fDy        = 2.f * fMaxDistance / (float) fNofBinsY;
  float fDr        = fMaxRadius / (float) fNofBinsR;
  int fNofBinsXY = fNofBinsX * fNofBinsY;

  void ht(std::vector<float> x, std::vector<float> y, std::vector<float> t) {
    
    std::vector<HoughHit> hits;
    for (int i = 0; i < x.size(); i++) {
      HoughHit hit;
      hit.x = x[i];
      hit.y = y[i];
      hit.t = t[i];
      hit.x2plusy2 = x[i]*x[i] + y[i]*y[i];
      hits.push_back(hit);
    }

    typedef std::vector<HoughHit>::iterator iH;

    std::vector<float> xc_vec;
    std::vector<float> yc_vec;
    std::vector<float> r_vec;
    std::vector<int> index0;
    std::vector<int> index1;
    std::vector<int> index2;
    //std::vector<std::vector<float>> result;

    float xcs, ycs;  // xcs = xc - fCurMinX
    float dx = 1.0f / fDx, dy = 1.0f / fDy, dr = 1.0f / fDr;

    for (iH iHit1 = hits.begin(); iHit1 != hits.end(); iHit1++) {
      float iH1X   = iHit1->x;
      float iH1Y   = iHit1->y;
      double time1 = iHit1->t;
      for (iH iHit2 = iHit1 + 1; iHit2 != hits.end(); iHit2++) {
        float iH2X   = iHit2->x;
        float iH2Y   = iHit2->y;
        double time2 = iHit2->t;
        if (std::fabs(time1 - time2) > fTimeCut) continue;

        float rx0 = iH1X - iH2X;  //rx12
        float ry0 = iH1Y - iH2Y;  //ry12
        float r12 = rx0 * rx0 + ry0 * ry0;

        if (r12 < fMinDistanceSq || r12 > fMaxDistanceSq) continue;

        float t10 = iHit1->x2plusy2 - iHit2->x2plusy2;
        for (iH iHit3 = iHit2 + 1; iHit3 != hits.end(); iHit3++) {
          float iH3X   = iHit3->x;
          float iH3Y   = iHit3->y;
          double time3 = iHit3->t;

          if (std::fabs(time1 - time3) > fTimeCut) continue;
          if (std::fabs(time2 - time3) > fTimeCut) continue;

          float rx1 = iH1X - iH3X;  //rx13
          float ry1 = iH1Y - iH3Y;  //ry13
          float r13 = rx1 * rx1 + ry1 * ry1;
          if (r13 < fMinDistanceSq || r13 > fMaxDistanceSq) continue;

          float rx2 = iH2X - iH3X;  //rx23
          float ry2 = iH2Y - iH3Y;  //ry23
          float r23 = rx2 * rx2 + ry2 * ry2;
          if (r23 < fMinDistanceSq || r23 > fMaxDistanceSq) continue;

          float det = rx2 * ry0 - rx0 * ry2;
          if (det == 0.0f) continue;
          float t19 = 0.5f / det;
          float t5  = iHit2->x2plusy2 - iHit3->x2plusy2;

          float xc = (t5 * ry0 - t10 * ry2) * t19;
          // xcs      = xc - fCurMinX; //FIXME:
          // int intX = int(xcs * dx);
          // if (intX < 0 || intX >= fNofBinsX) continue;

          float yc = (t10 * rx2 - t5 * rx0) * t19;
          // ycs      = yc - fCurMinY; //FIXME:
          // int intY = int(ycs * dy);
          // if (intY < 0 || intY >= fNofBinsY) continue;

          //radius calculation
          float t6 = iH1X - xc;
          float t7 = iH1Y - yc;
          //if (t6 > fMaxRadius || t7 > fMaxRadius) continue;
          float r = sqrt(t6 * t6 + t7 * t7);
          //if (r < fMinRadius) continue;
          //int intR = int(r * dr);
          //if (intR < 0 || intR >= fNofBinsR) continue;
          xc_vec.push_back(xc);
          yc_vec.push_back(yc);
          r_vec.push_back(r);
          index0.push_back(iHit1 - hits.begin());
          index1.push_back(iHit2 - hits.begin());
          index2.push_back(iHit3 - hits.begin());          
        }  // iHit1
      }    // iHit2
    }      // iHit3
    // print xc_vec
    result.push_back(xc_vec);
    result.push_back(yc_vec);
    result.push_back(r_vec);
    indices.push_back(index0);
    indices.push_back(index1);
    indices.push_back(index2);
    //return result;
  }

};

PYBIND11_MODULE(ht, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring

  //m.def("ht", &ht, "Hough transform");
  py::class_<HoughHit>(m, "HoughHit")
    .def(py::init<>())
    .def_readwrite("x", &HoughHit::x)
    .def_readwrite("y", &HoughHit::y)
    .def_readwrite("t", &HoughHit::t);
  py::class_<HoughTransform>(m, "HoughTransform")
    .def(py::init<std::vector<float>, std::vector<float>, std::vector<float>>())
    .def_readwrite("result", &HoughTransform::result)
    .def_readwrite("indices", &HoughTransform::indices);
}