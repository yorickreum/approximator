(* ::Package:: *)

BeginPackage["Kiessling`"]

KiesslingPlotParams::usage = "A PlotStyle for JMU lab.";
KiesslingHistogramParams::usage = "Histogram parameters/options for JMU lab.";
frameLegendFunction::usage = "A FrameLegendFunction for JMU lab."
CrossMarker::usage = "A cross for plots";
restylePlot::usage = "Restyle plots after their creation.";

Begin["`Private`"]

CrossMarker = Graphics[
{
Thickness[0.15],
Line[{{-1,-1},{1,1}}], Line[{{-1,1},{1,-1}}]
}, 
PlotRangePadding->None
];

KiesslingPlotParams = {
PlotTheme->"Detailed",
(* Thickness\[Rule] Thickness[0.001], *)
Frame->True,
FrameStyle->Directive[Black,18], (* frame border, labels text size *)
PlotMarkers->{CrossMarker,3},
GridLines->Automatic,
ImageSize -> Large
};

KiesslingHistogramParams = {
PlotTheme->"Detailed",
(* Thickness\[Rule] Thickness[0.001], *)
Frame->True,
FrameStyle->Directive[Black,14] (* frame border, labels text size *)
};

frameLegendFunction[legend_]:=Framed[legend,FrameStyle->Black,RoundingRadius->0,FrameMargins->5,Background->White]
 
restylePlot[plot_Graphics, styles_List, op : OptionsPattern[Graphics]] :=
 Module[{x = styles}, Show[
   MapAt[# /. {__, ln__Line} :> {Directive @ Last[x = RotateLeft@x], ln} &, plot, 1],
   op
]]

End[]

EndPackage[]
