set encoding iso_8859_15
set terminal postscript enhanced solid color "AvantGarde-Book" 20
set output "output_band.ps"
 
set key off
xscale=       1.0
xshift=0.0
xmin=       0.000000
xmax=       11.5887
set xrange [xmin*xscale-xshift:xmax*xscale-xshift]
ymin=      -1.000000
ymax=      1.000000
set yrange [ymin:ymax]
set border lw 2
eref=0.0
fact=1.0
gfact=1.0
point_size=1.0
color_red="red"
color_green="green"
color_blue="blue"
color_cyan="cyan"
color_magenta="magenta"
color_gold="gold"
color_pink="pink"
color_black="black"
color_olive="olive"
color_brown="brown"
color_gray="gray"
color_light_blue="light-blue"
color_orange="orange"
color_yellow="yellow"
band_lw=2
lshift=(ymax - ymin)/35.
shift=-(ymax - ymin)/40.
# set xlabel "k ({\305}^{-1})"
set ylabel "Energy (eV)"
eref=      7.6133
unset xtics
unset border
mark1 = 0.0000000000000000
mark2 = 0.5020978999999999
mark3 = 4.0909095999999998 
mark4 = 4.2691797214532601 
mark5 = 7.9346984368658067 
mark6 = 11.588736447189792 

set arrow from      mark1*xscale-xshift,ymin to       mark1*xscale-xshift,ymax nohead front lw   2 lc rgb color_black
set arrow from      mark2*xscale-xshift,ymin to       mark2*xscale-xshift,ymax nohead front lw   2 lc rgb color_black
set arrow from       mark1*xscale-xshift, ymin to       mark2*xscale-xshift,ymin nohead front lw   2 lc rgb color_black
set arrow from       mark1*xscale-xshift, ymax to       mark2*xscale-xshift,ymax nohead front lw   2 lc rgb color_black
set arrow from       mark1*xscale-xshift, 0.0 to       mark2*xscale-xshift,0.0 nohead front lw   2 lc rgb color_black
set arrow from      mark2*xscale-xshift,ymin to       mark2*xscale-xshift,ymax nohead front lw   2 lc rgb color_black
set arrow from      mark3*xscale-xshift,ymin to       mark3*xscale-xshift,ymax nohead front lw   2 lc rgb color_black
set arrow from       mark2*xscale-xshift, ymin to       mark3*xscale-xshift,ymin nohead front lw   2 lc rgb color_black
set arrow from       mark2*xscale-xshift, ymax to       mark3*xscale-xshift,ymax nohead front lw   2 lc rgb color_black
set arrow from       mark2*xscale-xshift, 0.0 to       mark3*xscale-xshift,0.0 nohead front lw   2 lc rgb color_black
set arrow from      mark3*xscale-xshift,ymin to       mark3*xscale-xshift,ymax nohead front lw   2 lc rgb color_black
set arrow from      mark4*xscale-xshift,ymin to       mark4*xscale-xshift,ymax nohead front lw   2 lc rgb color_black
set arrow from       mark3*xscale-xshift, ymin to       mark4*xscale-xshift,ymin nohead front lw   2 lc rgb color_black
set arrow from       mark3*xscale-xshift, ymax to       mark4*xscale-xshift,ymax nohead front lw   2 lc rgb color_black
set arrow from       mark3*xscale-xshift, 0.0 to       mark4*xscale-xshift,0.0 nohead front lw   2 lc rgb color_black
set arrow from      mark4*xscale-xshift,ymin to       mark4*xscale-xshift,ymax nohead front lw   2 lc rgb color_black
set arrow from      mark5*xscale-xshift,ymin to       mark5*xscale-xshift,ymax nohead front lw   2 lc rgb color_black
set arrow from       mark4*xscale-xshift, ymin to       mark5*xscale-xshift,ymin nohead front lw   2 lc rgb color_black
set arrow from       mark4*xscale-xshift, ymax to       mark5*xscale-xshift,ymax nohead front lw   2 lc rgb color_black
set arrow from       mark4*xscale-xshift, 0.0 to       mark5*xscale-xshift,0.0 nohead front lw   2 lc rgb color_black
set arrow from      mark5*xscale-xshift,ymin to       mark5*xscale-xshift,ymax nohead front lw   2 lc rgb color_black
set arrow from      mark6*xscale-xshift,ymin to       mark6*xscale-xshift,ymax nohead front lw   2 lc rgb color_black
set arrow from       mark5*xscale-xshift, ymin to       mark6*xscale-xshift,ymin nohead front lw   2 lc rgb color_black
set arrow from       mark5*xscale-xshift, ymax to       mark6*xscale-xshift,ymax nohead front lw   2 lc rgb color_black
set arrow from       mark5*xscale-xshift, 0.0 to       mark6*xscale-xshift,0.0 nohead front lw   2 lc rgb color_black

set label "{/Symbol G}" at       mark1*xscale-xshift, ymin + shift center
set label "Z" at       mark2*xscale-xshift, ymin + shift center
set label "P_1" at       mark3*xscale-xshift, ymin + shift center
set label "F" at       mark4*xscale-xshift, ymin + shift center
set label "{/Symbol G}" at       mark5*xscale-xshift, ymin + shift center
set label "L" at       mark6*xscale-xshift, ymin + shift center

plot  "bands.dat.gnu" u ($1*xscale-xshift):($2*fact-eref)*gfact w l lw band_lw lc rgb color_red
