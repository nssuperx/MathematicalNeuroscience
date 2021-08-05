#include "bm.hpp"
using namespace std;

namespace BMModule {
    enum class weight_mode : char {
        zero,
        random5,
        x36
    };

    void generate_weight(weight_mode wm){
        random_device rnd;                     // 非決定的な乱数生成器を生成, /dev/randomとかを見たりする. シード値の代わりに使う．
        mt19937 mt(rnd()); 
        vector<vector<double>> w;
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
}