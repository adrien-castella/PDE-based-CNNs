User tools:
	We start by giving an introduction on how to use the code to run simulations.
	We explain this by going through each file one by one and discussing the output. Then
	we give a few tips on key parts in the code that can be useful to play around with.

	First notice that each file should be run through command prompt. When opening command
	prompt use the command "cd PATH" to navigate to the folder location. Another option is
	to run the code through a console available in for example VisualStudio. (This does not
	require navigating to the correct path)
	
	Following this the code can be run through the command "py NAME.py". With the exception
	of only a few files, most require parameters to be set. The available parameters can be
	checked through the help command "py NAME.py -h".





	configurations.py
		Configurations files are an important component for running simulations. The
		configurations.py code is used to create new configurations. It has 2 key required
		parameters. Namly, the name set through "-n NAME" abd the type set through
		"-t TYPE". The type indicates which network to use. There are 3 different types,
		namely a conventional CNN ("CON") and 2 PDE-based CNNs ("GEN" and "DE").
		

		CONVENTIONAL CNN

		The conventional "CON" type is used for conventional CNN simulations. These
		configurations can only specify a number of channels used. Some other parameters
		such as the number of runs "-c", the epochs "-e", the learning rate "-r", and the
		multiplier "-g" can be modified through the system parameters as described by the
		help function. These same system parameters can be set regardless of the
		indicated type.
		
		
		GENERAL PDE-BASED CNN

		The general type "GEN" is used for simulations which use the same module
		configuration across every layer of the network. For example, if the diffusion
		module is used in layer 1, it is used in the remaining layers as well.
		
		
		DILATION AND EROSION PDE-BASED CNN

		The dilation/erosion type "DE" is used for specific configurations which use
		convection, dilation, and erosion. It allows a choice of using convection on
		every layer or on none. For dilation and erosion more specific choices are
		given through a binary notation and some system parameters.



		REMAINING SYSTEM PARAMETERS

		The system parameters "-f" and "-l" are used for the "DE" model to determine
		which module among dilation or erosion is used on the first and the last layer
		respectively. Only 1 module can be indicated.

		
		OTHER IMPORTANT DETAILS
		
		Many of the options for the configurations are not determined through the system
		parameters. When indicating the system parameters and pressing ENTER to run the
		code, a set of questions will be asked with regards to the configurations. Each
		question required a list (ex. [1, 2, 4, 4]) or a set (ex. {1, 2, 4}) to be
		indicated. Ensure to use the correct type which will be indicated after the
		default list or set.


			(UN)WANTED SET
			
			The last choices that the configurations file will request are the
			(un)wanted sets. It will first request to indicate either "0" or "1" to
			use the complement or the set itself. The set required is a list of
			binary values. These are only used for the types "GEN" and "DE" and both
			require different kinds of binary values. We will cover both now

			"GEN"
			The configuration in terms of the modules for the GEN model is defined by
			a binary number of length 4 for each available module (convection,
			dilation, erosion, and diffusion). Taking for example the binary number
			"1100", this indicates the configuration using convection and dilation
			only.

			"DE"
			The binary numbers indicate the modules for the DE model as well. However,
			they do not hold the same exact meaning as for the GEN model. As the
			convection module is already set previously, the binary number concerns
			only the convifguration of dilation and erosion in the 3 middle layers
			(layers 2, 3, and 4).
			The binary numbers have a length of 6. Each pair of binaries in this
			number indicates the configuration on one layer. Take for example the
			binary number "110011". This indicates that dilation + erosion (11) are
			used on the first of the middle layers and the last. However, the remaining
			layer does not use dilation or erosion. Another example is "111011". This
			uses dilation on each of the middle layers, but excludes erosion from
			layer 3 only (10).

			HOW
			These configurations need to be listed in a set.
			For the DE model an example would be to give
				{111011, 110011, 101010}
			For the GEN model an example would be to give
				{1100, 0110, 1001}

		
		FINAL DETAILS
		
		The configuration file essentially takes the cartesian product of each of the
		lists or sets that have been indicated and then creates a configuration file
		with the indicated name containing the resulting list which can be used for
		running simulations with "main.py".





	combine.py
		This code is created a supplement for the configurations file. It is used to
		concatinate two configuration files. This is used since the configurations.py
		code always takes the entire cartesian product. combine.py allows for different
		configuration lists.
		
		
		SYSTEM PARAETERS
		
		This file takes two main system parameters, namely the configuration files to be
		combined through "-c CONFIG1 CONFIG2 ..." and the name of the output
		configuration "-n NAME". Assuming existing configuration files "name_1", "name_2",
		and "name_3" an example would be
			py combine.py -c name_1 name_2 name_3 -n output
		which gives a configuration called "output" which is the union of the other 3.





	show_config.py
		This is the final supplement for configurations.py and combined.py. This code
		can be used to check the available configuration files and the details of these
		files.

		By giving no system parameters, the code simply outputs the list of names of the
		available configuration files. To get details on one of these files, the code
		must be run again with a system parameter set through "-n NAME". This of course
		requires the name of the configuration to be checked. The result is a list of
		dictionaries printed in the console.





	main.py
		This file can be seen as the most important component of the project. It is used
		to run the simulations indicated by configuration files.


		SYSTEM PARAMETERS
		
		Two key parameters are used for this file. The configuration, which can be set
		through "-c NAME". This should indicate the name of a previously created
		configuration file. Then the folder name, set through "-f FOLDER". This indicates
		the name of the folder which will be used for this simulation. The folder will be
		stored in the output folder which exists in the same directory as the code.

		OTHER PARAMETERS

		The remaining parameters are not required and details on them can be found
		through the help function "-h". We give additional details on 2 of these
		parameters.
		- For some old configuration files, no model type (ex. "GEN") is specified. These
		  configurations require the user to specify the model when running the main.py
		  code. This does not apply to newely created configurations.
		- The second parameter is the noise parameter denoted "-n". The code for adding
		  noise to the input images is under the comment "Adding noise to images". Under
		  the if statement two for loops are given to add noise to each image in the
		  training and testing sets respectively.

		OUTPUT
		
		Files:
			- "arguments"
				This file contains the system parameters specified for this run
			- "conf_code"
				This file is a copy of the configuration file used for this run
			- "listed_details"
				This file contain a list of tripplets which contain the max DICE,
				accuracy, and AUC score over each epoch of a configuration. These
				are given in the same order as the configurations in "conf_code".
			- "max_index"
				This file contains the max score for each metric together with the
				configuration that achieved it.

		Folders:
			- "images"
				Contains plots of each of the metrics across the epochs for each
				configuration. The names correspond to the configuration names.
			- "times"
				Contains JSON files which give the running time for each epoch.
				Each configuration has two respective files for the testing and
				training time respectively.
			- "JSON file"
				Contains the JSON files giving an overview for each configuration.
				A single file contains the number of images used for training, the
				max DICE, accuracy, and AUC scores achieved together with the
				respective epoch. Additionally, the number of parameters used in
				the network and the average testing and training times per epoch
				are given.
			- Other folders
				These other folders each have the name of a configuration. They
				store the model after each epoch. These can be loaded again through
				PyTorch and be used for further investigation. They are also used
				when running the "plot_out.py" code.

	


	
	plot_out.py
		This code is used to plot the kernels of a given model and the output for a
		specified testing image.

		
		GETTING STARTED
		
		Recall the model files generated by main.py for each epoch of each configuration.
		These files can be used for "plot_out.py" by copying a model for a specific epoch
		into the destination "input/models". When the file is present in this folder it
		can be used to plot the mentioned details into a folder whose name is specified
		and will be stored in the "test" folder.


		REQUIRED PARAMETERS
		
		The requried parameters are the name of the model file "-n NAME", the ouput
		folder name "-f FOLDER" and the model type "-m TYPE" (ex. "GEN"). The name "-n"
		must be identical to the name given to the file copied to "input/models".

		The remaining parameters can be found through the help function "-h".

		
		OUTPUT COLOR CODING
		
		The kernel images are heatmaps. The higher the value the brighter the color.
		The network output is color coded according to the True Positives (Green), True
		Negatives (White), False Positives (Red), and False Negatives (Blue).






Folder structure:
	We've already covered the folders "test" and  "output". The folder "input" folder contains
	three sub-folders. The folder "configurations" contains the JSON files for the configurations
	generated by configurations.py. The "drive-full" folder contains the training and testing
	data. The "models" folder has already been covered in plot_out.py.




Required packages:
	All of the following packages may be installed through either pip or conda in command line

	PyTorch - Required. Find instructions on https://pytorch.org/
	Argparse - Required. As of Python >= 2.7 it is included within the standard library.
	Matplotlib - Requried. Find instructions on https://pypi.org/project/matplotlib/
	Torchvision - Required. Find instructions on https://pypi.org/project/torchvision/
	Scikit-learn - Required. Find instructions on https://scikit-learn.org/stable/install.html
	Numpy - Required. Find instructions on https://numpy.org/install/
	Seaborn - Required. Find instructions on https://seaborn.pydata.org/installing.html
	Libtiff - Recommended. Problems may occur when importing .tif images if this package is not
		  used.