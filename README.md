# MPPT-using-AI-Techniques
Refer to nnmodelTrainingsetmeasure.m as the master file
This project is directed towards the advancement of Hybrid Solar Electric Vehicles.


MPPT is done using a stacked ensemble Neural Network Model, which takes:
Ambient Temperature
Solar Irradiation (DNI)
Solar Irradiation (DHI)
Solar Altitude Angle
Solar Azimuth Angle

Data
Weather Data was collected from 49 American Cities' TMY weather data stored in a Database managed by NREL, called the NSRDB 
(National Solar Radiation Database)
Next, the data was processed through NREL's SAM (System Advisory Model) Application, to generate the corresponding Maximum Power Point
operating voltage and current through their built in models, for a generic solar module whose dimensions are suitable to be fit on the 
roof of a standard vehicle (or to be set as standalone system on residential buildings, various locomotives, etc).

Data Augmentation:
TO capture variuos non linearities, the data was augmented as follows
Suppose x is input parameter.
append: 
x^2
x^3
sin(x)
cos(x)

After which , the data was normalized.


NN training.
The model utilizes a stacked ensemble Neural Network model, which trains on 2 levels.

Level0:


function mapping from input to output. 
Operating Voltage and Current were trained on two separate, but identical Neural Network Models
Activation Function? None
Backpropagation method? fmincg
                        
                        Minimize a continuous differentialble multivariate function
                        Copyright ,Carl Edward Rasmussen,2002.
                        
                       
                        The Polack-Ribiere flavour of conjugate gradients is used to compute search directions,
                        and a line search using quadratic and cubic polynomial approximations and the
                        Wolfe-Powell stopping criteria is used together with the slope ratio method
                        for guessing initial step sizes
                        

Level 1
function mapping from output of level 1(corresponding to a fresh dataset) to the ground truth (Y , normalized).
This layer acts as a fine tuning model, to improve results from layer 0.


Further developments will be appended as and when developed and tested.
