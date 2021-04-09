#include<iostream>
#include<vector>
#include<algorithm>
#include<numeric>
using namespace std;

int main(){
    const int n = 100;
    vector<vector<int>> A(n, vector<int>(n));
    vector<int> x(n);
    vector<int> y(n,0);
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            A.at(i).at(j) = i + j + 1;
        }
    }

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            cout << A.at(i).at(j) << ' ';
        }
        cout << endl;
    }

    cout << endl;

    for(int i=0; i<n; i++){
        x.at(i) = i;
        cout << x.at(i) << ' ';
    }

    cout << endl;

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            y.at(i) += A.at(i).at(j) * x.at(j);
        }
    }

    for(int i=0; i<n; i++){
        cout << y.at(i) << ' ';
    }

    cout << endl;
    return 0;
}