set terminal pdfcairo color enhanced font "Helvetica,24" size 12in, 9in
set output "result01.pdf"
set xlabel "FPR"
set ylabel "CDR"
# set xrange[0:0.2]
# set yrange[0:1.1]
# set nokey

set datafile separator ","
plot "pgm_sg.csv" using 1:2 title "Sg" with linespoints linestyle 1, "pgm_st.csv" using 1:2 title "St" with linespoints linestyle 2, "pgm_sp.csv" using 1:2 title "Sp" with linespoints linestyle 3

set terminal x11
replot