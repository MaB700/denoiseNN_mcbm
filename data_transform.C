#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "TFile.h"
#include "TTree.h"

int data_transform(){ //FIX:

    TFile* inFile = new TFile("./data.root", "READ");
    TTree* inTree = (TTree*)inFile->Get("train");
    float inTime[2304];
    float inTarget[2304];
    inTree->SetBranchAddress("time", inTime);
    inTree->SetBranchAddress("tar", inTarget);

    TFile* outFile = new TFile("./data_txy.root", "RECREATE");
    outFile->cd();
    TTree* outTree = new TTree("data", "data");
    std::vector<float> outTime;
    std::vector<float> outX;
    std::vector<float> outY;
    std::vector<float> outTarget;
    outTree->Branch("time", &outTime);
    outTree->Branch("x", &outX);
    outTree->Branch("y", &outY);
    outTree->Branch("tar", &outTarget);

    for(int i{}; i < inTree->GetEntries(); i++){
        inTree->GetEntry(i);
        // do stuff with inTime and inTarget
    }




    return 0;
}