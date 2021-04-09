set terminal pdfcairo color enhanced font "Helvetica,24" size 12in, 9in
set output "result01.pdf"
set xlabel "Time"
# set ylabel "D.C."
set xrange[0:]
set yrange[0:1.1]
set nokey

row = system("cat am.csv | awk \'NR==1 {c += (split($1, a, \",\")-1)} END{print c}\'") # 列数を気合で数える．
set datafile separator ","
plot for [i = 2:row:1] "am.csv" using 1:i with linespoints linestyle i

set terminal x11
replot