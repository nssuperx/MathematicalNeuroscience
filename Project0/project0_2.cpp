#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
using namespace std;

vector<double> dot(vector<vector<double>> T, vector<double> x){
    vector<double> Y(T.size());
    for(int i=0; i<T.size(); ++i){
        for(int j=0; j<x.size(); ++j){
            Y.at(i) += (double)T.at(i).at(j) * x.at(j);
        }
    }
    return Y;
}

double eucledean_norm(vector<double> array){
    double num = 0;
    for(double a: array){
        num += a*a;
    }
    return sqrt(num);
}

vector<double> div_vector(vector<double> numerator, vector<double> denominator){
    if(numerator.size() != denominator.size()){
        cout << "div_vector: can't division!!!" << endl;
        exit(EXIT_FAILURE);
    }
    vector<double> div(numerator.size());
    for(int i=0; i<numerator.size(); ++i){
        div.at(i) = numerator.at(i) / denominator.at(i);
    }
    return div;
}

int main(){
    vector<vector<double>> T = {{6.0, -3.0, -7.0}, {-1.0, 2.0, 1.0}, {5.0, -3.0, -6.0}};
    for(vector<double> t: T){
        for(double num: t){
            cout << num << ' ';
        }
        cout << endl;
    }

    vector<vector<double>> X(51, vector<double>(3));
    X.at(0).at(0) = 4.0;
    X.at(0).at(1) = 0.0;
    X.at(0).at(2) = 3.0;

    for(int i=1; i<=50; ++i){
        vector<double> Tx = dot(T, X.at(i-1));
        double TxNorm = eucledean_norm(Tx);
        for(int j=0; j<X.at(i).size(); ++j){
            X.at(i).at(j) = Tx.at(j) / TxNorm;
        }
    }

    ofstream output("project0_2.csv");
    for(vector<double> x: X){
        for(double num: x){
            output << num << ',';
        }
        output << endl;
    }
    output.close();
}
