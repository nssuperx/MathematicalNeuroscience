#include <iostream>
#include <bitset>
using namespace std;

int main(){
    bitset<4> bs(1);
    cout << bs << endl;
    cout << bs.to_ulong() << endl;

    bs = bitset<4>(3);
    cout << bs << endl;
    cout << bs.to_ulong() << endl;
    return 0;
}