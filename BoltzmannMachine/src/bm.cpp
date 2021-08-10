#include "bm.hpp"
using namespace std;

int map_accumulate(map<string, int> m);
double calc_KLdiv(vector<double> q, vector<double> p);

void q1a();
void q1b();
void q1c();
void q2h();

class BoltzmannMachine {
  public:
    enum class weight_mode : char {
        zero,
        random5,
        x36
    };

    BoltzmannMachine(int neuron, weight_mode wm);
    void set_default_state();
    void set_default_weight(weight_mode wm);
    void set_dist_q(vector<double> dist);
    vector<double> get_dist_p();
    void update_state(int neuron_number);
    void update_weight();
    void count_state();
    void reset_state_count();
    vector<double> calc_stationary_dist();
    vector<int> get_state_count();
    void print_weight();
    void count_frequency(vector<int> x, map<string, int> &freq);
    void count_frequency(vector<int> x, vector<vector<int>> &count_matrix);
    void reset_frequency();
    void generate_x();
    void set_model_frec(vector<vector<double>> freq_matrix, int batch_size);
    void print_frequency(int state_num);

  private:
    vector<vector<double>> w;
    vector<int> x;
    vector<int> state_count;

    int neuron;
    const double T = 1.0;
    const double lr = 0.01; //Learning Rate
    int batch_size = 100;

    random_device rnd;                     // 非決定的な乱数生成器を生成, /dev/randomとかを見たりする. シード値の代わりに使う．
    mt19937 mt;                            // mersenne twister 32bit 擬似乱数生成器
    uniform_int_distribution<> rand2;      // [0, 1] 範囲の一様乱数を生成
    uniform_real_distribution<> rand_real; // [0.0, 1.0)の実数一様分布

    vector<vector<int>> same_one_count_f;
    vector<vector<int>> same_one_count_g;

    vector<double> dist_q; // xを生成するときの確率分布
};

BoltzmannMachine::BoltzmannMachine(int neuron, weight_mode wm = weight_mode::zero) {
    cout << "Boltzmann Machine" << endl;
    x.resize(neuron + 1);
    w.resize(x.size(), vector<double>(x.size()));
    same_one_count_f.resize(x.size(), vector<int>(x.size()));
    same_one_count_g.resize(x.size(), vector<int>(x.size()));
    state_count.resize((int)pow(2, neuron));

    mt = mt19937(rnd());
    rand2 = uniform_int_distribution<>(0, 1);
    rand_real = uniform_real_distribution<>(0.0, 1.0);

    for(int i = 0; i < x.size(); i++) {
        for(int j = 0; j < x.size(); j++) {
            same_one_count_f.at(i).at(j) = 0;
            same_one_count_g.at(i).at(j) = 0;
        }
    }

    cout << "Set default state." << endl;
    set_default_state();

    cout << "Set weight." << endl;
    set_default_weight(wm);
    cout << "Done setup." << endl;
}

void BoltzmannMachine::set_default_state() {
    for(int i = 0; i < x.size(); i++) {
        x.at(i) = 0;
    }
    x.at(0) = 1;
}

void BoltzmannMachine::set_default_weight(weight_mode wm) {
    if(wm == weight_mode::zero) {
        for(int i = 0; i < w.size(); i++) {
            for(int j = 0; j < w.at(i).size(); j++) {
                w.at(i).at(j) = 0.0;
            }
        }
    } else if(wm == weight_mode::random5) {
        uniform_real_distribution<> rand_real5(-5, 5);
        for(int i = 0; i < w.size(); i++) {
            for(int j = i + 1; j < w.at(i).size(); j++) {
                w.at(i).at(j) = rand_real5(mt);
                w.at(j).at(i) = w.at(i).at(j);
            }
            w.at(i).at(i) = 0.0;
        }
    } else if(wm == weight_mode::x36) {
        for(int i = 0; i < w.size(); i++) {
            for(int j = 0; j < w.at(i).size(); j++) {
                w.at(i).at(j) = 0.0;
            }
        }

        vector<int> x3{1, 0, 1, 1};
        for(int i = 0; i < x3.size(); i++) {
            for(int j = i + 1; j < x3.size(); j++) {
                w.at(i).at(j) += (2 * x3.at(i) - 1) * (2 * x3.at(j) - 1);
            }
        }

        vector<int> x6{1, 1, 1, 0};
        for(int i = 0; i < x6.size(); i++) {
            for(int j = i + 1; j < x6.size(); j++) {
                w.at(i).at(j) += (2 * x6.at(i) - 1) * (2 * x6.at(j) - 1);
            }
        }

        for(int i = 0; i < w.size(); i++) {
            for(int j = i + 1; j < w.at(i).size(); j++) {
                w.at(j).at(i) = w.at(i).at(j);
            }
            w.at(i).at(i) = 0.0;
        }
    }
}

void BoltzmannMachine::set_dist_q(vector<double> dist) {
    dist_q = dist;
}

vector<double> BoltzmannMachine::get_dist_p() {
    vector<double> dist_p(state_count.size());
    for(int i = 0; i < state_count.size(); i++) {
        dist_p.at(i) = (double)state_count.at(i) / (double)batch_size;
    }
    return dist_p;
}

void BoltzmannMachine::update_state(int neuron_number) {
    count_state();
    count_frequency(x, same_one_count_g);
    double u = 0.0;
    for(int i = 0; i < x.size(); i++) {
        u += w.at(neuron_number).at(i) * (double)x.at(i);
    }
    double prob = 1.0 / (1.0 + exp(-u / T));
    x.at(neuron_number) = prob < rand_real(mt) ? 0 : 1;
}

void BoltzmannMachine::update_weight() {
    for(int i = 0; i < w.size() - 1; i++) {
        for(int j = i + 1; j < w.at(i).size(); j++) {
            w.at(i).at(j) += lr * (double)(same_one_count_f.at(i).at(j) - same_one_count_g.at(i).at(j)) / (double)batch_size;
            w.at(j).at(i) = w.at(i).at(j);
        }
    }
}

void BoltzmannMachine::reset_state_count() {
    for(int i = 0; i < state_count.size(); i++) {
        state_count.at(i) = 0;
    }
}

vector<double> BoltzmannMachine::calc_stationary_dist() {
    // TODO: 再帰使うべき
    vector<double> all_E((int)pow(2, x.size() - 1));
    vector<int> tmp_x(x.size());
    tmp_x.at(0) = 1;
    vector<int> x_elements{0, 1};
    int state_number_count = 0;
    double c = 0.0;
    for(int x1 : x_elements) {
        for(int x2 : x_elements) {
            for(int x3 : x_elements) {
                tmp_x.at(1) = x1;
                tmp_x.at(2) = x2;
                tmp_x.at(3) = x3;
                double tmp_E = 0.0;
                for(int i = 0; i < tmp_x.size() - 1; i++) {
                    for(int j = i + 1; j < tmp_x.size(); j++) {
                        tmp_E -= w.at(i).at(j) * (double)tmp_x.at(i) * (double)tmp_x.at(j);
                    }
                }
                c += exp(-tmp_E / T);
                all_E.at(state_number_count) = tmp_E;
                state_number_count++;
            }
        }
    }

    vector<double> p(all_E.size());
    // 答えの代入と表示をいっしょにやってる
    for(int i = 0; i < all_E.size(); i++) {
        p.at(i) = exp(-all_E.at(i) / T) / c;
        cout << "x" << i << " "
             << "stationary_dist: " << p.at(i) << endl;
    }

    return p;
}

void BoltzmannMachine::count_state() {
    // shellのchmodみたいな考え方でやってる
    int state_number = 0;
    for(int i = 0; i < x.size() - 1; i++) {
        state_number += x.at(x.size() - 1 - i) * (int)pow(2, i);
    }
    state_count.at(state_number) += 1; // いまの状態のカウント TODO: もっとスマートにできんかな
}

vector<int> BoltzmannMachine::get_state_count() {
    return state_count;
}

void BoltzmannMachine::print_weight() {
    cout << "weight" << endl;
    for(int i = 0; i < w.size(); i++) {
        for(int j = 0; j < w.at(i).size(); j++) {
            cout << w.at(i).at(j) << "  ";
        }
        cout << endl;
    }
}

void BoltzmannMachine::generate_x() {
    vector<int> x_from_q(x.size());
    for(int i = 0; i < batch_size; i++) {
        double dist_accumulate = 0;
        double rand_value = rand_real(mt);
        int q_idx;
        for(q_idx = 0; q_idx < dist_q.size(); q_idx++) {
            dist_accumulate += dist_q.at(q_idx);
            if(rand_value < dist_accumulate) {
                break;
            }
        }
        // 以下，気合で生成
        switch(q_idx) {
        case 0:
            x_from_q = {1, 0, 0, 0};
            break;
        case 1:
            x_from_q = {1, 0, 0, 1};
            break;
        case 2:
            x_from_q = {1, 0, 1, 0};
            break;
        case 3:
            x_from_q = {1, 0, 1, 1};
            break;
        case 4:
            x_from_q = {1, 1, 0, 0};
            break;
        case 5:
            x_from_q = {1, 1, 0, 1};
            break;
        case 6:
            x_from_q = {1, 1, 1, 0};
            break;
        case 7:
            x_from_q = {1, 1, 1, 1};
            break;
        default:
            break;
        }
        count_frequency(x_from_q, same_one_count_f);
    }
}

void BoltzmannMachine::set_model_frec(vector<vector<double>> freq_matrix, int batch_size) {
    if(same_one_count_f.size() != freq_matrix.size() || same_one_count_f.at(0).size() != freq_matrix.at(0).size()) {
        cout << "wrong freq matrix size!!" << endl;
        return;
    }
    for(int i = 0; i < same_one_count_f.size(); i++) {
        for(int j = 0; j < same_one_count_f.at(i).size(); j++) {
            same_one_count_f.at(i).at(j) = (int)(freq_matrix.at(i).at(j) * (double)batch_size);
        }
    }
}

void BoltzmannMachine::count_frequency(vector<int> x, map<string, int> &freq) {
    // TODO: なにこれ
    for(int i = 0; i < x.size() - 1; i++) {
        for(int j = i + 1; j < x.size(); j++) {
            if(x.at(i) == 1 && x.at(j) == 1) {
                freq[to_string(i) + to_string(j)] += 1;
            }
        }
    }
}

void BoltzmannMachine::count_frequency(vector<int> x, vector<vector<int>> &count_matrix) {
    for(int i = 0; i < x.size() - 1; i++) {
        for(int j = i + 1; j < x.size(); j++) {
            if(x.at(i) == 1 && x.at(j) == 1) {
                count_matrix.at(i).at(j) += 1;
            }
        }
    }
}

void BoltzmannMachine::reset_frequency() {
    for(int i = 0; i < x.size(); i++) {
        for(int j = 0; j < x.size(); j++) {
            same_one_count_g.at(i).at(j) = 0;
        }
    }
}

void BoltzmannMachine::print_frequency(int state_num) {
    // TODO: 設計ミスった．
    cout << "frequency f" << endl;
    for(int i = 0; i < same_one_count_f.size(); i++) {
        for(int j = 0; j < same_one_count_f.at(i).size(); j++) {
            cout << same_one_count_f.at(i).at(j) << "  ";
        }
        cout << endl;
    }

    cout << "frequency g" << endl;
    if(state_num == 0) return;
    for(int i = 0; i < same_one_count_g.size(); i++) {
        for(int j = 0; j < same_one_count_g.at(i).size(); j++) {
            cout << same_one_count_g.at(i).at(j) << "  ";
        }
        cout << endl;
    }
}

int map_accumulate(map<string, int> m) {
    int sum = 0;
    for(auto itr = m.begin(); itr != m.end(); ++itr) {
        sum += itr->second;
    }
    return sum;
}

double calc_KLdiv(vector<double> q, vector<double> p) {
    double D = 0.0;
    for(int i = 0; i < q.size(); i++) {
        D += q.at(i) * log(q.at(i) / (p.at(i) + 0.00001));
    }
    return D;
}

vector<vector<double>> make_freq_f(int neuron) {
    vector<vector<double>> freq_f;
    freq_f.resize(neuron + 1, vector<double>(neuron + 1));
    for(int i = 0; i < freq_f.size(); i++) {
        for(int j = 0; j < freq_f.size(); j++) {
            freq_f.at(i).at(j) = 0;
        }
    }
    freq_f.at(0).at(1) = 0.7;
    freq_f.at(0).at(2) = 0.6;
    freq_f.at(0).at(3) = 0.35;
    freq_f.at(1).at(2) = 0.5;
    freq_f.at(1).at(3) = 0.2;
    freq_f.at(2).at(3) = 0.15;
    return freq_f;
}

int main() {
    q2h();
    return 0;
}

void q1c() {
    int neuron = 3;
    BoltzmannMachine bm(neuron, BoltzmannMachine::weight_mode::x36);

    random_device rnd;                         // 非決定的な乱数生成器を生成, /dev/randomとかを見たりする. シード値の代わりに使う．
    mt19937 mt(rnd());                         // mersenne twister 32bit 擬似乱数生成器
    uniform_int_distribution<> rand_int(1, 3); // [1, 3] 範囲の一様乱数を生成

    int l = 1000;
    cout << "l=" << l << endl;

    bm.print_weight();

    for(int i = 0; i < l; i++) {
        bm.update_state(rand_int(mt));
    }
}

void q2h() {
    int neuron = 3;
    BoltzmannMachine bm(neuron, BoltzmannMachine::weight_mode::zero);

    random_device rnd;                         // 非決定的な乱数生成器を生成, /dev/randomとかを見たりする. シード値の代わりに使う．
    mt19937 mt(rnd());                         // mersenne twister 32bit 擬似乱数生成器
    uniform_int_distribution<> rand_int(1, 3); // [1, 3] 範囲の一様乱数を生成

    int l = 1000;
    cout << "l=" << l << endl;

    vector<double> dist_q{0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.4, 0.1};
    vector<double> dist_p;
    vector<vector<double>> freq_f = make_freq_f(neuron);
    int batch_size = 100;
    bm.set_dist_q(dist_q);
    bm.set_model_frec(freq_f, batch_size);
    // bm.generate_x();
    vector<double> D(l);
    for(int i = 0; i < l; i++) {
        bm.reset_frequency();
        bm.reset_state_count();
        for(int j = 0; j < batch_size; j++) {
            bm.update_state(rand_int(mt));
        }
        dist_p = bm.get_dist_p();
        D.at(i) = calc_KLdiv(dist_q, dist_p);
        bm.update_weight();
    }

    bm.print_frequency(batch_size);
    bm.print_weight();

    bm.reset_state_count();
    for(int i = 0; i < l; i++) {
        bm.update_state(rand_int(mt));
    }

    vector<int> state_count = bm.get_state_count();

    for(int i = 0; i < state_count.size(); i++) {
        cout << "x" << i << " : " << state_count.at(i) << '\t' << (double)state_count.at(i) / (double)l << endl;
    }

    bm.calc_stationary_dist();

    ofstream outfile("KLdiv.csv");
    for(int i = 0; i < D.size(); i++) {
        // if(i % 100 == 0) cout << i << ":\t\t" << D.at(i) << endl;
        outfile << i << ',' << D.at(i) << endl;
    }
    outfile.close();
}
