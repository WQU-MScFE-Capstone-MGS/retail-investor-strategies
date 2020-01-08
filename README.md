# Introduction
This repo contains the implementation of the WorldQuant University's Capstone project submitted by:
1. Mikhail Shishlenin
2. Ganesh Harke and
3. Suresh Koppisetti

This project is accomplished by extending the QuantConnect's framework to implement a smart beta algorithm. The source code can be found under the <code>src</code> folder. Below is the structure of the code organization:

<pre>
src
|__ run_config.py
|__ main.py
|__ fundamental_data.py
|__ algo_type.py
</pre>

As mentioned above, as part of the research for our project, we have used the QuantConnect platform. QuantConnect provides its framework to write our own algorithms either on the cloud at https://www.quantconnect.com/terminal/ or by downloading and using the QuantConnect's Lean engine locally. In either case, the implementation remains pretty much the same. We performed most of our analysis on the cloud because of the availablity of large US equity datasets. We have also used the local Lean engine to test the custom dollar bars we generated usign the tick data we obtained for some stocks on Russian stock exchange.


# Implementation
As part of our research we have decided to implement a Smart Beta alogorithm. Below is the class diagram if of the implementation:

![Class Diagram](images/cs_class_diagram.png)
