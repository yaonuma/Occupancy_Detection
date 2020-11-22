# Occupancy_Detection


Abstract: 

Experimental data used for binary classification (room occupancy) from Temperature,Humidity,Light and CO2. Ground-truth occupancy was obtained from time stamped pictures that were taken every minute.


Source: 

Luis Candanedo, luismiguel.candanedoibarra '@' umons.ac.be, UMONS.


Data Set Information: 

Three data sets are submitted, for training and testing. Ground-truth occupancy was obtained from time stamped pictures that were taken every minute.


Attribute Information:

date time year-month-day hour:minute:second
Temperature, in Celsius
Relative Humidity, %
Light, in Lux
CO2, in ppm
Humidity Ratio, Derived quantity from temperature and relative humidity, in kgwater-vapor/kg-air
Occupancy, 0 or 1, 0 for not occupied, 1 for occupied status


Relevant Papers:

Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models. Luis M. Candanedo, VÃ©ronique Feldheim. Energy and Buildings. Volume 112, 15 January 2016, Pages 28-39.


Analysis Summary:

The sensor data is treated as time series data for 1D convolutional neural network. Although event sequence is not intuitively considered necessary
for the described classification problem, in other words, a simpler logistic regression approach may produce similar levels of accuracy, this 
serves useful framework for more complex problems, such as those involving more complex signals or more sensors. Of particular interest to me
would be structural health monitoring in automotive or aerospace applications, where it would be necessary to know the realtime structrual integrity
of a mechanical system before catastrophic failure.
