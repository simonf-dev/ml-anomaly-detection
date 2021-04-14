cd tmpdir


lineCount(file)= system('wc -l  < '.file.' ')
getValue(row,col,filename) = system('awk ''{if (NR == '.row.') print $'.col.'}'' '.filename.'')
node_count = words(node_hostnames)
set terminal svg enhanced font 'verdana' background rgb 'white' size 1600,(600 *node_count)
set multiplot layout (node_count),2  title '{/:Bold {/=20 Beaker job '.beaker_job_id.' performance metrics}}'
set style fill transparent solid 0.2
set grid
set key font ',8'
set datafile missing "None"
set y2tics
set xlabel 'Time [UTC]'
set xtics rotate 90

max_percent_storage=word(max_y,1)
max_size_storage=word(max_y,2)
cpu_max=word(max_y,3)
swaps_max=word(max_y,4)


first_group_colours='#B40066 #0A2B97 #FFD600 #FE6B72 #8B4513 #5CF700'
second_group_colours='#FFA500 #00A925 #E01500 #E853B4 #000000 #4C34F8'
all_group_colours='#FFA500 #B40066 #0A2B97 #00A925 #FFD600 #E01500 #FE6B72 #8B4513 #5CF700 #E853B4 #000000 #4C34F8'
color_cpu_steal = '#B40066'
color_cpu_ksmd = '#FFA500'
color_mem_used = '#0A2B97'
color_cpu_sys = '#00A925'
color_cpu_user = '#FFD600'
color_cpu_idle = '#E01500'
color_swap_pages = '#006868'
set xdata time
set format x '%H:%M:%S'
set border behind

do for [i=1:node_count] {

    thr_cpu_idle  = getValue(1,1,word(threshold_files,(i*4)-1))
    thr_cpu_steal = getValue(1,2,word(threshold_files,(i*4)-1))
    thr_cpu_sys = getValue(1,3,word(threshold_files,(i*4)-1))
    thr_cpu_user = getValue(1,4,word(threshold_files,(i*4)-1))
    thr_cpu_ksmd = getValue(1,5,word(threshold_files,(i*4)-1))
    thr_mem_used = getValue(1,6,word(threshold_files,(i*4)-1))
    thr_swap_pages = getValue(1,1,word(threshold_files,i*4))

    set title '{/:Bold {/=16 Host '.word(node_hostnames,i).'}}' offset 30
    unset ylabel
    unset yrange
    unset y2range
    unset y2label

    if (lineCount(word(plot_files,(i*4)-1) ) ne "1" && lineCount(word(plot_files,i*4)) ne "1") {
        set ylabel '[%]'
        set yrange [0:cpu_max*100]
        set y2range [0:swaps_max]
        set y2label '[units]'
	plot \
	word(plot_files,(i*4)-1) using ($1):($2*100) with lines axes x1y1 lw 1.5 lt rgb (color_cpu_idle) title 'cpu.idle [%]',\
	'' using ($1):($3*100) with lines axes x1y1 lw 1.5 lt rgb (color_cpu_steal) title 'cpu.steal [%]',\
	'' using ($1):($4*100) with lines axes x1y1 lw 1.5 lt rgb (color_cpu_sys) title 'cpu.sys [%]',\
	'' using ($1):($5*100) with lines axes x1y1 lw 1.5 lt rgb (color_cpu_user) title 'cpu.user [%]',\
	'' using ($1):($6*100) with lines axes x1y1 lw 1.5 lt rgb (color_cpu_ksmd) title 'ksmd_utime+proctime [%]',\
        '' using ($1):($7*100) with lines axes x1y1 lw 1.5 lt rgb (color_mem_used) title 'mem.available [%]',\
        '' using ($1):($2*100) with filledcurves below y1=thr_cpu_idle*100 axes x1y1 lt rgb (color_cpu_idle) notitle,\
	'' using ($1):($3*100) with filledcurves above y1=thr_cpu_steal*100 axes x1y1 lt rgb (color_cpu_steal) notitle,\
    	'' using ($1):($4*100) with filledcurves above y1=thr_cpu_sys*100 axes x1y1 lt rgb (color_cpu_sys) notitle,\
    	'' using ($1):($5*100) with filledcurves above y1=thr_cpu_user*100 axes x1y1 lt rgb (color_cpu_user) notitle,\
    	'' using ($1):($6*100) with filledcurves above y1=thr_cpu_ksmd*100 axes x1y1 lt rgb (color_cpu_ksmd) notitle,\
        '' using ($1):($7*100) with filledcurves below y1=thr_mem_used*100 axes x1y1 lt rgb (color_mem_used) notitle,\
    	word(plot_files,i*4)  using ($1):($2) with lines axes x1y2 lw 1.5 lt rgb (color_swap_pages) title 'swap.pagesout [units]',\
        '' using ($1):($2) with filledcurves above y2=thr_swap_pages axes x1y2 lt rgb (color_swap_pages) notitle,

    } else {
        if ((lineCount(word(plot_files,(i*4)-1)) ne "1")  &&  lineCount(word(plot_files,i*4)) eq "1"){
	    set ylabel '[%]'
            set yrange [0:cpu_max*100]

	        plot \
	    word(plot_files,(i*4)-1) using ($1):($2) with lines axes x1y1 lt rgb (color_cpu_idle) title 'cpu.idle [%]',\
	    '' using ($1):($3*100) with lines axes x1y1 lt rgb (color_cpu_sys) title 'cpu.sys [%]',\
            '' using ($1):($4*100) with lines axes x1y1 lt rgb (color_cpu_user) title 'cpu.user [%]',\
            '' using ($1):($5*100) with lines axes x1y1 lt rgb (color_cpu_ksmd) title 'ksmd_utime+proctime [%]',\
            '' using ($1):($6*100) with lines axes x1y1 lt rgb (color_mem_used) title 'mem.available [%]',\
            '' using ($1):($2*100) with filledcurves below y1=thr_cpu_idle axes x1y1 lt rgb (color_cpu_idle) notitle,\
            '' using ($1):($3*100) with filledcurves above y1=thr_cpu_sys axes x1y1 lt rgb (color_cpu_sys) notitle,\
            '' using ($1):($4*100) with filledcurves above y1=thr_cpu_user axes x1y1 lt rgb (color_cpu_user) notitle,\
            '' using ($1):($5*100) with filledcurves above y1=thr_cpu_ksmd axes x1y1 lt rgb (color_cpu_ksmd) notitle,\
            '' using ($1):($6*100) with filledcurves below y1=thr_mem_used axes x1y1 lt rgb (color_mem_used) notitle,\

	} else {
	    if ((lineCount(word(plot_files,(i*4)-1)) eq "1")  &&  lineCount(word(plot_files,i*4)) ne "1") {
	        set yrange [0:swaps_max]
        	set ylabel '[units]'
	        	plot \
	        word(plot_files,i*4) using ($1):($2) with lines axes x1y1 lt rgb (color_swap_pages) title 'swap.pagesout [units]',\
                '' using ($1):($2) with filledcurves above y1=thr_swap_pages axes x1y1 lt rgb (color_swap_pages) notitle
            } else {
	        plot 0 with lines
            }
        }
    }

    set title ' '
    unset ylabel
    unset yrange
    unset y2range
    unset y2label

    unset xdata
    if (lineCount(word(plot_files,(i*4)-3)) ne "1") {
        stats word(plot_files,(i*4)-3) prefix 'disk' output
    }
    if (lineCount(word(plot_files,(i*4)-2)) ne "1") {
        stats word(plot_files,(i*4)-2) prefix 'storage' output
    }
    set xdata time



    if (lineCount(word(plot_files,(i*4)-3)) ne "1" && lineCount(word(plot_files,(i*4)-2)) ne "1") {
        set ylabel '[%]'
    	set yrange [0:max_percent_storage]
    	set y2range [0:max_size_storage]
    	set y2label '[kB]'
	plot for [k=2:disk_columns] (word(plot_files,(i*4)-3)) using ($1):k with lines axes x1y1 lt (word(first_group_colours,k)) title columnhead, \
        for [k=2:disk_columns] (word(plot_files,(i*4)-3)) using ($1):k with filledcurves above y1=getValue(1,k-1,word(threshold_files,(i*4)-3)) axes x1y1 lt rgb (word(first_group_colours,k)) notitle, \
        for [j=2:storage_columns] (word(plot_files,(i*4)-2)) using ($1):j with lines axes lt rgb (word(second_group_colours,j)) x1y2 title columnhead,\
	for [j=2:storage_columns] (word(plot_files,(i*4)-2)) using ($1):j with filledcurves above y2=getValue(1,k-1,word(threshold_files,(i*4)-2))) axes x1y2 lt rgb (word(second_group_colours,j)) notitle
    } else {
        if ((lineCount(word(plot_files,(i*4)-3)) ne "1")  &&  lineCount(word(plot_files,(i*4)-2)) eq "1"){
                set ylabel '[%]'
        	set yrange [0:max_percent_storage]
		plot for [k=2:disk_columns] (word(plot_files,(i*4)-3)) using ($1):k with lines axes x1y1 lt rgb (word(all_group_colours,k)) title columnhead, \
		for [k=2:disk_columns] (word(plot_files,(i*4)-3)) using ($1):k with filledcurves above y1=getValue(1,k-1,word(threshold_files,(i*4)-3)) axes x1y1 lt rgb (word(all_group_colours,k)) notitle
        } else {
                if ((lineCount(word(plot_files,(i*4)-3)) eq "1")  &&  lineCount(word(plot_files,(i*4)-2)) ne "1") {
                    set yrange [0:max_size_storage]
   		    set ylabel '[kB]'
		    plot for [k=2:storage_columns] word(plot_files,(i*4)-2)) using ($1):k with lines axes x1y1 lt rgb (word(all_group_colours,k)) title columnhead, \
		    for [k=2:storage_columns] (word(plot_files,(i*4)-2)) using ($1):k with filledcurves above y1=getValue(1,k-1,word(threshold_files,(i*4)-3))) axes x1y1 lt rgb (word(all_group_colours,k)) notitle
                } else {
                    plot 0 with lines

                }
        }
    }

}
