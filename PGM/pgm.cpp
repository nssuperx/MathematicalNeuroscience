#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>
using namespace std;

class pgm {
  private:
    const double sigma = 1.0;

    map<int, map<int, double>> px; // NOTE: vectorでもいい気がする
    vector<int> x;
    vector<vector<double>> y;
    mt19937 mt; // mersenne twister 32bit 擬似乱数生成器
    uniform_real_distribution<> rand_x;
    normal_distribution<> rand_y;

    double p00, p01, p10, p11; // データ生成時に使用
                               // vector<double> mp;

  public:
    pgm();
    double compute_s_god();
    double compute_s_template();
    double compute_s_parts();
    void generate_data(int xi, int n1, int n2);
    int judge(double value, double theta);
    void calc(int alpha, int xi, int n1);
    void out_result(vector<pair<double, double>> &result, string filename);
    void print_x_y();
};

pgm::pgm() {
    px[0][0] = 0.6;
    px[0][1] = 0.1;
    px[1][0] = 0.1;
    px[1][1] = 0.2;
    random_device seed; // 非決定的な乱数生成器を生成, /dev/randomとかを見たりする. シード値の代わりに使う．
    mt = mt19937(seed());
    rand_x = uniform_real_distribution<>(0.0, 1.0); // [0.0, 1.0] 範囲の一様乱数を生成
    rand_y = normal_distribution<>(0.0, 1.0);       // 平均0.0, 標準偏差1.0の正規分布

    // データ生成時に使用
    p00 = px[0][0];
    p01 = px[0][0] + px[0][1];
    p10 = px[0][0] + px[0][1] + px[1][0];
    p11 = 1.0; // 統一させるために一応（未使用）

    // TODO: このへんたぶん違う、解決するまで残しとく
    // mp.resize(px.size());
    // for(int i=0; i<mp.size(); i++){
    //     double sum = 0;
    //     for(int j=0; j<px.at(i).size(); j++){
    //         sum += px[i][j];
    //     }
    //     mp.at(i) = sum;
    // }
}

double pgm::compute_s_god() {
    double sum = 0.0;
    for(int x1 = 0; x1 <= 1; x1++) {
        for(int x2 = 0; x2 <= 1; x2++) {
            double expsum = 0.0;
            for(int i = 0; i < 2; i++) {
                for(int j = 0; j < y.at(i).size(); j++) {
                    expsum += (1 - 2 * y.at(i).at(j) + 2 * y.at(i).at(j) * px[i][i] - px[i][i] * px[i][i]) / 2 * sigma * sigma;
                }
            }
            sum += px[x1][x2] * exp(expsum);
        }
    }

    // return 1 / (sum / px[1][1]);
    return px[1][1] / sum;
}

double pgm::compute_s_template() {
    double expsum = 0.0;
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < y.at(i).size(); j++) {
            expsum += (1 - 2 * y.at(i).at(j)) / 2 * sigma * sigma;
        }
    }

    return 1 / (1 + px[0][0] / px[1][1] * exp(expsum));
}

double pgm::compute_s_parts() {
    // NOTE: 拡張性が全くないので何とかする
    double expsum = 0.0;
    for(int j = 0; j < y.at(0).size(); j++) {
        expsum += (1 - 2 * y.at(0).at(j)) / 2 * sigma * sigma;
    }
    double pr1 = 1 / (1 + (px[0][0] + px[0][1]) / (px[1][0] + px[1][1]) * exp(expsum));

    expsum = 0.0;
    for(int j = 0; j < y.at(1).size(); j++) {
        expsum += (1 - 2 * y.at(1).at(j)) / 2 * sigma * sigma;
    }
    double pr2 = 1 / (1 + px[1][0] / px[1][1] * exp(expsum));

    return pr1 * pr2;
}

void pgm::generate_data(int xi, int n1, int n2) {
    double r = rand_x(mt);
    if(r < p00) {
        x.at(0) = 0;
        x.at(1) = 0;
    } else if(r < p01) {
        x.at(0) = 0;
        x.at(1) = 1;
    } else if(r < p10) {
        x.at(0) = 1;
        x.at(1) = 0;
    } else {
        x.at(0) = 1;
        x.at(1) = 1;
    }

    for(int i = 0; i < xi; i++) {
        for(int j = 0; j < n1; j++) {
            y.at(i).at(j) = (double)x.at(i) + rand_y(mt);
        }
    }
}

int pgm::judge(double value, double theta) {
    if(value > theta) {
        return 1;
    } else {
        return 0;
    }
}

void pgm::calc(int alpha, int xi, int n1) {
    vector<pair<double, double>> roc_sg, roc_st, roc_sp; // NOTE: 横軸fpr, 縦軸cdr
    pair<double, double> p_sg, p_st, p_sp;

    x.resize(xi);
    y.resize(xi, vector<double>(n1));

    // いろんなカウンタ
    int xa_not11;
    int xa_is11;

    int fpr_count_Sg;
    int cdr_count_Sg;
    int fpr_count_St;
    int cdr_count_St;
    int fpr_count_Sp;
    int cdr_count_Sp;

    // 初めの1回、(0.0, 0.0)をプロットするため。
    p_sg = make_pair(0.0, 0.0);
    p_st = make_pair(0.0, 0.0);
    p_sp = make_pair(0.0, 0.0);

    roc_sg.push_back(p_sg);
    roc_st.push_back(p_st);
    roc_sp.push_back(p_sp);

    // いろんな方法で計算
    for(double theta = 0.0; theta <= 1.0; theta += 0.01) {
        xa_is11 = 0;
        xa_not11 = 0;

        fpr_count_Sg = 0;
        cdr_count_Sg = 0;
        fpr_count_St = 0;
        cdr_count_St = 0;
        fpr_count_Sp = 0;
        cdr_count_Sp = 0;

        for(int i = 0; i < alpha; i++) {
            generate_data(xi, n1, n1);
            // print_x_y();
            if(x.at(0) == 1 && x.at(1) == 1) {
                xa_is11++;
                cdr_count_Sg += judge(compute_s_god(), theta);
                cdr_count_St += judge(compute_s_template(), theta);
                cdr_count_Sp += judge(compute_s_parts(), theta);
            } else {
                xa_not11++;
                fpr_count_Sg += judge(compute_s_god(), theta);
                fpr_count_St += judge(compute_s_template(), theta);
                fpr_count_Sp += judge(compute_s_parts(), theta);
            }
        }

        p_sg.first = (double)fpr_count_Sg / (double)xa_not11;
        p_sg.second = (double)cdr_count_Sg / (double)xa_is11;
        p_st.first = (double)fpr_count_St / (double)xa_not11;
        p_st.second = (double)cdr_count_St / (double)xa_is11;
        p_sp.first = (double)fpr_count_Sp / (double)xa_not11;
        p_sp.second = (double)cdr_count_Sp / (double)xa_is11;

        roc_sg.push_back(p_sg);
        roc_st.push_back(p_st);
        roc_sp.push_back(p_sp);
    }

    sort(roc_sg.begin(), roc_sg.end());
    sort(roc_st.begin(), roc_st.end());
    sort(roc_sp.begin(), roc_sp.end());

    out_result(roc_sg, "pgm_sg.csv");
    out_result(roc_st, "pgm_st.csv");
    out_result(roc_sp, "pgm_sp.csv");
}

void pgm::out_result(vector<pair<double, double>> &result, string filename) {
    FILE *out_file;
    out_file = fopen(filename.c_str(), "w");

    for(pair<double, double> p : result) {
        fprintf(out_file, "%8.7lf,%8.7lf\n", p.first, p.second);
    }

    fclose(out_file);
}

void pgm::print_x_y() {
    for(int i = 0; i < x.size(); i++) {
        cout << 'X' << i << ": " << x.at(i) << '\t';
    }
    cout << endl;

    for(int i = 0; i < y.size(); i++) {
        for(int j = 0; j < y.at(i).size(); j++) {
            cout << 'Y' << i << j << ": " << y.at(i).at(j) << '\t';
        }
    }
    cout << endl;
}

int main() {
    pgm model = pgm();
    int alpha = 1000;
    int n1 = 3;
    int i = 2; // xの上付き数字 i
    model.calc(alpha, i, n1);
    return 0;
}