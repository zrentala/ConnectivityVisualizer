Introduction:
For my undergrad [senior thesis](https://7ac0abc7-e08a-4934-9d60-2f8d93f73017.filesusr.com/ugd/7af63b_94b83b118a7b456093385915c47fa018.pdf), I compared the neurological effects of action observation therapy when performed by human therapists versus our lab's social robot. The main feature we used to compare was functional connectivity (FC), which measures the statistical dependency and synchronized activity between spatially separated brain regions. Unforuntately, there were no packages to display the FC metrics in an easy way.

Thus, I wanted to build an app that can aid researchers in creating visually appealing FC plots for papers and gives them an easy way to analyze basic FC metrics for an initial discovery. 

While I was able to create a simple prototype of the app, key decisions made at the beginning of the project hampered the performance of the app and made it not worth maintianing. This project log details the my design process, problems and optimizaitons, and what I learned from the mistakes along the way.

Design: 
While planning the app, I had three key principles in mind.

1. Flexibility. Users should be able to controls every single component in the plot: the color, size, and curve of each node and edge, different type sof plots 2d topoographical map, 3d model, heatmap, the number of edges on screen. 

2. Modularity. Like any good software, I wanted each aspect of the code to be siloed away.

3. Insigntful. Adding threshold and graph analysis which are standard practice in FC and network neurosceince analysis. Listed stats, total node, and edges, visibile edges, coenction densitym efficiency, modularity, strength.

I chose plotly as the main grpahical engine because it was already made for creating flexible plots, had prebuilt aspects and was interactive.

Graph of app structures here:

Problems:
When implementing the app, I soon realized that the goals I set was causing issues.

Flexibility: while controlling every aspect is good, it forces me to draw every aspect on screen which greatly slows down the app. EEG montages often include 60 - 300 electodes. The number of total electrodes is (equation here) or geneal number Every action to edit the graph was O(n^2). Optimize by limiting number of items that had to be edited. If the color chnaged the we edit every vsibile edge, if threhsold chnage, compare cached mask and only edit edges whose visbility differs, if node color chanegs then only change node color. I vectorized calculations and bacthced the updates

I attempted to mutlithread/multiprocess these clauclations and updates but plotly is locked and cannot parallelize. 

What I learned:


