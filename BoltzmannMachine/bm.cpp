#include<iostream>
#include<vector>
#include<random>
#include<cmath>
using namespace std;


class BoltzmannMachine{
    public:
    enum class weight_mode : char{
        zero,
        random5
    };

    BoltzmannMachine(int neuron, weight_mode wm);
    void set_default_state();
    void set_default_weight(weight_mode wm);
    void update_state(int neuron_number);
    void count_state();
    vector<double> calc_stationary_dist();
    vector<int> get_state_count();

    private:
    vector<vector<double>> w;
    vector<int> x;
    vector<int> state_count;

    int neuron;
    const double T = 1.0;

    random_device rnd;      // 非決定的な乱数生成器を生成, /dev/randomとかを見たりする. シード値の代わりに使う．
    mt19937 mt;             // mersenne twister 32bit 擬似乱数生成器
    uniform_int_distribution<> rand2;               // [0, 1] 範囲の一様乱数を生成
    uniform_real_distribution<> rand_real;          // [0.0, 1.0)の実数一様分布

};

BoltzmannMachine::BoltzmannMachine(int neuron, weight_mode wm = weight_mode::zero){
    cout << "Boltzmann Machine" << endl;
    x.resize(neuron + 1);
    w.resize(x.size(), vector<double>(x.size()));
    state_count.resize((int)pow(2,neuron));

    mt = mt19937(rnd());
    rand2 = uniform_int_distribution<>(0, 1);
    rand_real = uniform_real_distribution<>(0.0, 1.0);

    cout << "Set default state." << endl;
    set_default_state();

    cout << "Set weight." << endl;
    set_default_weight(wm);
    cout << "Done setup." << endl;
}

void BoltzmannMachine::set_default_state(){
    for(int i=0; i<x.size(); i++){
        x.at(i) = 0;
    }
    x.at(0) = 1;
}

void BoltzmannMachine::set_default_weight(weight_mode wm){
    if(wm == weight_mode::zero){
        for(int i=0; i<w.size(); i++){
            for(int j=0; j<w.at(i).size(); j++){
                w.at(i).at(j) = 0;
            }
        }
    }else if(wm == weight_mode::random5){
        uniform_int_distribution<> rand_int(-5, 5);
        for(int i=0; i<w.size(); i++){
            for(int j=0; j<w.at(i).size(); j++){
                w.at(i).at(j) = rand_int(mt);
            }
        }
    }
}

void BoltzmannMachine::update_state(int neuron_number){
    count_state();
    double u = 0.0;
    for(int i=0; i<x.size(); i++){
        u += w.at(neuron_number).at(i) * x.at(i);
    }
    double prob = 1.0 / (1.0 + exp(-u / T));
    x.at(neuron_number) = prob < rand_real(mt) ? 0 : 1;
}

vector<double> BoltzmannMachine::calc_stationary_dist(){
    // TODO: 再帰使うべき
    vector<double> all_E((int)pow(2,x.size()-1));
    vector<int> all_x(x.size());
    vector<int> x_elements{0,1};
    int state_number_count = 0;
    double c = 0.0;
    for(int x1: x_elements){
        for(int x2: x_elements){
            for(int x3: x_elements){
                all_x.at(1) = x1;
                all_x.at(2) = x2;
                all_x.at(3) = x3;
                double tmp_E = 0.0;
                for(int i=0; i<all_x.size(); i++){
                    for(int j=i; j<all_x.size(); j++){
                        tmp_E -= w.at(i).at(j) * (double)all_x.at(i) * (double)all_x.at(j);
                    }
                }
                c += exp(-tmp_E / T);
                // TODO: なんとかしないと
                all_E.at(state_number_count) = tmp_E;
                state_number_count++;
            }
        }
    }

    vector<double> p(all_E.size());
    // 答えの代入と表示をいっしょにやってる
    for(int i=0; i<all_E.size(); i++){
        p.at(i) = exp(-all_E.at(i) / T) / c;
        cout << "x" << i+1 << " " << "stationary_dist: " << p.at(i) << endl;
    }

    return p;
}

void BoltzmannMachine::count_state(){
    // shellのchmodみたいな考え方でやってる
    int state_number = 0;
    for(int i=0; i<x.size()-1; i++){
        state_number += x.at(x.size()-1-i) * (int)pow(2,i);
    }
    state_count.at(state_number) += 1;      // いまの状態のカウント TODO: もっとスマートにできんかな
}

vector<int> BoltzmannMachine::get_state_count(){
    return state_count;
}

int main(){
    // TODO: bitsetで実装するべき
    // https://cpprefjp.github.io/reference/bitset/bitset.html
    // https://cpprefjp.github.io/reference/bitset/bitset/to_ullong.html
    int neuron = 3;
    BoltzmannMachine bm(neuron, BoltzmannMachine::weight_mode::zero);

    random_device rnd;      // 非決定的な乱数生成器を生成, /dev/randomとかを見たりする. シード値の代わりに使う．
    mt19937 mt(rnd());             // mersenne twister 32bit 擬似乱数生成器
    uniform_int_distribution<> rand_int(1, 3);        // [0, 1] 範囲の一様乱数を生成

    int l = 1000000;
    for(int i=0; i<l; i++){
        bm.update_state(rand_int(mt));
    }

    vector<int> state_count = bm.get_state_count();

    // int sum_debug = 0;
    for(int i=0; i<state_count.size(); i++){
        cout << "x" << i+1 << " : " << state_count.at(i) << endl;
        // sum_debug += state_count.at(i);
    }
    // cout << sum_debug << endl;

    bm.calc_stationary_dist();

    return 0;
}

