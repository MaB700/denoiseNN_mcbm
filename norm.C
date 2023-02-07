//#include <cmath>
#include <iostream>
#if !defined(__CLING__)
#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TLegend.h"
#include "TColor.h"
#include "TStyle.h"
#endif

void norm(){

    TFile* f = new TFile("/home/martin/lustre/mrich_denoisenn/data/resultsTb_0.root", "READ");
    TCanvas* c = new TCanvas();
    c->cd();
    auto h = (TH2D*)f->Get("fhRadiusToTRingHits");
    // normalize rows
    // for (int i = 1; i <= h->GetNbinsX(); i++) {
    //     double sum = 0;
    //     for (int j = 1; j <= h->GetNbinsY(); j++) {
    //         sum += h->GetBinContent(i, j);
    //     }
    //     for (int j = 1; j <= h->GetNbinsY(); j++) {
    //         h->SetBinContent(i, j, h->GetBinContent(i, j) / sum);
    //     }
    // }

    for (int i = 1; i <= h->GetNbinsY(); i++) {
        double sum = 0.001;
        for (int j = 1; j <= h->GetNbinsX(); j++) {
            sum += h->GetBinContent(j, i);
        }
        for (int j = 1; j <= h->GetNbinsX(); j++) {
            h->SetBinContent(j, i, h->GetBinContent(j, i) / sum);
        }
    }
    
    h->Draw("colz");
    c->SaveAs("./radiusToT.pdf");


}