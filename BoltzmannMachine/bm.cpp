#include<iostream>
#include<vector>
#include<random>
#include<cmath>
#include<algorithm>
#include<numeric>
#include<map>
#include<string>
using namespace std;


int map_accumulate(map<string, int> m);

class BoltzmannMachine{
    public:
    enum class weight_mode : char{
        zero,
        random5,
        x36
    };

    BoltzmannMachine(int neuron, weight_mode wm);
    void set_default_state();
    void set_default_weight(weight_mode wm);
    void update_state(int neuron_number);
    void count_state();
    vector<double> calc_stationary_dist();
    vector<int> get_state_count();
    void print_weight();
    void count_frequency(vector<int> x, map<string, int>& freq);
    void generate_x();
    void print_frequency();

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

    map<string, int> freq_count_f;
    map<string, int> freq_count_g;

};

BoltzmannMachine::BoltzmannMachine(int neuron, weight_mode wm = weight_mode::zero){
    cout << "Boltzmann Machine" << endl;
    x.resize(neuron + 1);
    w.resize(x.size(), vector<double>(x.size()));
    state_count.resize((int)pow(2,neuron));

    mt = mt19937(rnd());
    rand2 = uniform_int_distribution<>(0, 1);
    rand_real = uniform_real_distribution<>(0.0, 1.0);

    /*
    freq_count_f.resize(((int)pow(2,neuron+1) - (neuron+1)) / 2);
    freq_count_g.resize(((int)pow(2,neuron+1) - (neuron+1)) / 2);

    for(int i=0; i<freq_count_f.size(); i++){
        freq_count_f.at(i) = 0;
        freq_count_g.at(i) = 0;
    }
    */

   for(int i=0; i<x.size()-1; i++){
        for(int j=i+1; j<x.size(); j++){
            freq_count_f[to_string(i) + to_string(j)] = 0;
            freq_count_g[to_string(i) + to_string(j)] = 0;
        }
    }

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
                w.at(i).at(j) = 0.0;
            }
        }
    }else if(wm == weight_mode::random5){
        uniform_real_distribution<> rand_real5(-5, 5);
        for(int i=0; i<w.size(); i++){
            for(int j=i+1; j<w.at(i).size(); j++){
                w.at(i).at(j) = rand_real5(mt);
                w.at(j).at(i) = w.at(i).at(j);
            }
            w.at(i).at(i) = 0.0;
        }
    }else if(wm == weight_mode::x36){
        for(int i=0; i<w.size(); i++){
            for(int j=0; j<w.at(i).size(); j++){
                w.at(i).at(j) = 0.0;
            }
        }

        vector<int> x3{1,0,1,1};
        for(int i=0; i<x3.size(); i++){
            for(int j=i+1; j<x3.size(); j++){
                w.at(i).at(j) += (2*x3.at(i) - 1) * (2*x3.at(j) - 1);
            }
        }

        vector<int> x6{1,1,1,0};
        for(int i=0; i<x6.size(); i++){
            for(int j=i+1; j<x6.size(); j++){
                w.at(i).at(j) += (2*x6.at(i) - 1) * (2*x6.at(j) - 1);
            }
        }

        for(int i=0; i<w.size(); i++){
            for(int j=i+1; j<w.at(i).size(); j++){
                w.at(j).at(i) = w.at(i).at(j);
            }
            w.at(i).at(i) = 0.0;
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
    vector<int> tmp_x(x.size());
    tmp_x.at(0) = 1;
    vector<int> x_elements{0,1};
    int state_number_count = 0;
    double c = 0.0;
    for(int x1: x_elements){
        for(int x2: x_elements){
            for(int x3: x_elements){
                tmp_x.at(1) = x1;
                tmp_x.at(2) = x2;
                tmp_x.at(3) = x3;
                double tmp_E = 0.0;
                for(int i=0; i<tmp_x.size()-1; i++){
                    for(int j=i+1; j<tmp_x.size(); j++){
                        tmp_E -= w.at(i).at(j) * (double)tmp_x.at(i) * (double)tmp_x.at(j);
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
        cout << "x" << i << " " << "stationary_dist: " << p.at(i) << endl;
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

void BoltzmannMachine::print_weight(){
    cout << "weight" << endl;
    for(int i=0; i<w.size(); i++){
        for(int j=0; j<w.at(i).size(); j++){
            cout << w.at(i).at(j) << "  ";
        }
        cout << endl;
    }
}

void BoltzmannMachine::generate_x(){
    int gen_num = 100;
    vector<int> x_from_q(x.size());
    vector<double> dist_q{0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.4, 0.1};
    for(int i=0; i<gen_num; i++){
        double dist_accumulate = 0;
        double rand_value = rand_real(mt);
        int q_idx;
        for(q_idx=0; q_idx<dist_q.size(); q_idx++){
            dist_accumulate += dist_q.at(q_idx);
            if(rand_value < dist_accumulate){
                break;
            }
        }
        // 以下，気合で生成
        switch (q_idx){
        case 0:
            x_from_q = {1,0,0,0};
            break;
        case 1:
            x_from_q = {1,0,0,1};
            break;
        case 2:
            x_from_q = {1,0,1,0};
            break;
        case 3:
            x_from_q = {1,0,1,1};
            break;
        case 4:
            x_from_q = {1,1,0,0};
            break;
        case 5:
            x_from_q = {1,1,0,1};
            break;
        case 6:
            x_from_q = {1,1,1,0};
            break;
        case 7:
            x_from_q = {1,1,1,1};
            break;
        default:
            break;
        }
        count_frequency(x_from_q, freq_count_f);
    }
}

void BoltzmannMachine::count_frequency(vector<int> x, map<string, int>& freq){
    // TODO: なにこれ
    for(int i=0; i<x.size()-1; i++){
        for(int j=i+1; j<x.size(); j++){
            if(x.at(i) == 1 && x.at(j) == 1){
                freq[to_string(i) + to_string(j)] += 1;
            }
        }
    }
}

void BoltzmannMachine::print_frequency(){
    // TODO: 設計ミスった．
    cout << "frequency f" << endl;
    int sum = map_accumulate(freq_count_f);
    for(auto itr = freq_count_f.begin(); itr != freq_count_f.end(); ++itr) {
        cout << "f" << itr->first << ":" << '\t' << itr->second << '\t' << (double)itr->second / (double)sum << endl;
    }

    cout << "frequency g" << endl;
    sum = map_accumulate(freq_count_g);
    if(sum == 0) return;
    for(auto itr = freq_count_g.begin(); itr != freq_count_g.end(); ++itr) {
        cout << "f" << itr->first << ":" << '\t' << itr->second << '\t' << (double)itr->second / (double)sum << endl;
    }
}

int map_accumulate(map<string, int> m){
    int sum = 0;
    for(auto itr = m.begin(); itr != m.end(); ++itr) {
        sum += itr->second;
    }
    return sum;
}

int main(){
    // TODO: bitsetで実装するべき
    // https://cpprefjp.github.io/reference/bitset/bitset.html
    // https://cpprefjp.github.io/reference/bitset/bitset/to_ullong.html
    int neuron = 3;
    BoltzmannMachine bm(neuron, BoltzmannMachine::weight_mode::x36);

    random_device rnd;      // 非決定的な乱数生成器を生成, /dev/randomとかを見たりする. シード値の代わりに使う．
    mt19937 mt(rnd());             // mersenne twister 32bit 擬似乱数生成器
    uniform_int_distribution<> rand_int(1, 3);        // [0, 1] 範囲の一様乱数を生成

    bm.print_weight();

    int l = 1000000;
    cout << "l=" << l << endl;
    for(int i=0; i<l; i++){
        bm.update_state(rand_int(mt));
    }

    vector<int> state_count = bm.get_state_count();

    // int sum_debug = 0;
    for(int i=0; i<state_count.size(); i++){
        cout << "x" << i << " : " << state_count.at(i) << '\t' << (double)state_count.at(i)/(double)l << endl;
        // sum_debug += state_count.at(i);
    }
    // cout << sum_debug << endl;

    bm.calc_stationary_dist();

    // bm.generate_x();
    // bm.print_frequency();

    return 0;
}

