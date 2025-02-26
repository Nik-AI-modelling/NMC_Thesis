#include <iostream>
#include <vector>
#include <fstream>

#include "TaylorGreenAnalytical.h"
#include "AdvectionVelocityCalc.h"
#include "Print_2Dvector.h"
#include "FilePaths.h"
#include <math.h>
#include <string>
#include "NN.h"

# define M_PI 3.14159265358979323846  /* pi */


int main()
{
	TaylorGreenAnalytical TaGrAn1;
	std::cout << "Velocity Ux " << TaGrAn1.getValueUx_TaylorGreenAna() << "\n";
	std::cout << "Velocity Uy " << TaGrAn1.getValueVy_TaylorGreenAna() << "\n";
	std::cout << "Velocity P " << TaGrAn1.getValueP_TaylorGreenAna() << "\n";



	//Calculate Taylor Green distribution. Reference data
	//The row counter
	int count = 0;
	std::vector<std::vector<float>> vect2D_ux_TGr;
	std::vector<std::vector<float>> vect2D_uy_TGr;
	std::vector<std::vector<float>> vect2D_p_TGr;
	std::vector<std::vector<float>> vect2D_x_TGr;
	std::vector<std::vector<float>> vect2D_y_TGr;

	//Generate matrix with calc results
	float T = 0.0;//Time 0 sec
	float RangeMax = 2*M_PI;
	float StepInRange = M_PI/10;
	int NumberRowsVector = RangeMax / StepInRange;
	//ScaleVectors
	vect2D_ux_TGr.resize(NumberRowsVector);
	vect2D_ux_TGr.resize(NumberRowsVector);
	vect2D_uy_TGr.resize(NumberRowsVector);
	vect2D_p_TGr.resize(NumberRowsVector);
	vect2D_x_TGr.resize(NumberRowsVector);
	vect2D_y_TGr.resize(NumberRowsVector);

	TaGrAn1.setValue_T(T);
	for (float i = 0.0; i < RangeMax;i += StepInRange) { //row
		for (float j = 0.0; j < RangeMax;j += StepInRange) { //column
			TaGrAn1.setValue_x(i);
			TaGrAn1.setValue_y(j);
			float Ux= TaGrAn1.getValueUx_TaylorGreenAna();
			float Uy= TaGrAn1.getValueVy_TaylorGreenAna();
			float P = TaGrAn1.getValueP_TaylorGreenAna();
		//	std::cout << "Velocity Uy " << j << "=" << TaGrAn1.getValueVy_TaylorGreenAna() << "\n";
		//	std::cout << "Velocity p " << "=" << TaGrAn1.getValueP_TaylorGreenAna() << "\n";
			vect2D_ux_TGr[count].push_back(Ux);
			vect2D_uy_TGr[count].push_back(Uy);
			vect2D_p_TGr[count].push_back(P);
			vect2D_x_TGr[count].push_back(i);
			vect2D_y_TGr[count].push_back(j);
		}
		//The row counter 
		count = count + 1;
	}
	//Print Teylor-Green Vortex vectors into external files

	//2D vector Ux
	Print_2Dvector print (vect2D_ux_TGr, "vect2D_ux_TGr");

	//2D vector Uy
	Print_2Dvector print2 (vect2D_uy_TGr, "vect2D_uy_TGr");
	//Print_2Dvector print(vect2D_p_TGr, "vect2D_p_TGr");

	//x vector 
	Print_2Dvector print3 (vect2D_x_TGr, "vect2D_x_TGr");

	//y vector
	Print_2Dvector print4 (vect2D_y_TGr, "vect2D_y_TGr");

	//Create an instance of the AdvectionVelocityCalc class 
	//This will create advected velocity based on the initial conditions from Teylor-Green vortex
	float dt = 0.1;
	float t_end = 1;
	//AdvectionVelocityCalc AdvAnalVel("TaG", &TaGrAn1, dt, t_end, RangeMax, StepInRange, NumberRowsVector);
	NN n({ 2,20,10,2 }, { "vx","vy" });

   

	//Load training data
	n.load(path_x_TGr, path_y_TGr, path_ux_TGr, path_uy_TGr);

	return 0;

	//Create an object of the neural network 
	//One input-coordinate -x,y 
	//Output
		//NN::NN({2,20,10,2}, { "vx","vy" });
		//int choise;
		//std::cout << "Please enter what you wish to do next (train NN-1 or test NN-2) ";
		//std::cin >> choise;

		//switch (choise) {
		//case 1:
		//	if (nn.data_index==dset_train.count())
		//	break;
		//case 2:

		//	break;
		//default:
		//	std::cout << "Invalid choise. \n";
		//}
}