intersections=[[1,0,1,0,1],[1,0,1,0,1,0]]
#Number the alpha and beta circles 0 to g-1. Number all the intersection points starting at 0, so that the points on alpha_i are before points on alpha_j for all i<j, and the intersections on each alpha-circle are numbered cyclically (orientation and starting point immaterial); the j-th point on alpha_i lies in beta_{intersections[i][j]}.

boundaries=[
    [0,1,10,9],
    [9,8,6,7,4,0],
    [5,6,8,7,0,4],
    [4,3,10,5],
    [1,0,7,6],
    [2,1,6,5],
    [3,4,7,8],
    [2,3,8,9],
    ]
#Number the regions (complement of alpha and beta), currently except the basepoint, starting at 0; boundaries[i] is a list of intersection points appearing on the boundary of the i-th region, cyclically ordered as per the orientation of the boundary. Furthermore, the arc joining the first two points in the list lies on some alpha-circle (we will partially check for this in the main program). Unfortunately, this means, from now on, we are assuming all regions are planar with one boundary (which can intersect itself).

abs_gradings_manual=False
#(chosen_generators,their_abs_gradings)=
#Absolute gradings. If you want to add abs_gradings manually, set abs_gradings_manual to True, uncomment the above line and add it after the equality sign (give a list of chosen generators (each generator is an ordered tuple of intersection points), one in each SpinC structure, and their absolute gradings). If not, set abs_gradings_manual to False and comment the above line. 

there_is_action=False
#image_of_intersections=
#Sometimes there is an action (of Z/2). If so, set there_is_action to True, uncomment the above line and give the images of the intersection points under this action as a list after the equality sign. If not, set there_is_action to False and comment the above line. 


#Main program
load hf-hat.sage


#Printing
#pretty_print()
#Use pretty_print function. You may input SpinC_structures as a list of SpinC structures to concentrate on (default is everything). If output_file specified, stores it; otherwise, displays it. You may provide interchanges as a list of interchanges of generators to make printing prettier. You may provide cancellations as a list of cancellations of generators to simplify the complex. (If there_is_action, try to do interchanges and cancellations equivariantly.)

#the complex in SpinC structure that was arbitrarily numbered 2
pretty_print(output_file='hf-hat-output-for-Sigma_2_3_4-SpinC_2-0.pdf',SpinC_structures=[2,])
#after the first round of cancellations
pretty_print(output_file='hf-hat-output-for-Sigma_2_3_4-SpinC_2-1.pdf',SpinC_structures=[2,],cancellations=[(8,2),(10,5)])