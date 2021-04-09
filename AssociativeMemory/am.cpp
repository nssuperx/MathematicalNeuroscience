#include<iostream>
#include<cstdio>
#include<string>
#include<vector>
#include<random>

const int N = 1000;
const int M = 80;

class amNet{
    public:
    amNet();
    double sgn(std::vector<double>& bx, int i);
    void flip_set_x(int m, int a);      //flipして想起？させるxを設定する．
    void run(int times, int flip_xa, std::vector<int>& flip_list);
    void update();
    double directon_cos();
    void out_result();                  //結果をcsvで書き出す．

    private:
    std::vector<std::vector<double>> xa;
    std::vector<std::vector<double>> w;
    std::vector<double> x;
    std::vector<std::vector<double>> result;

    int m;      // flipするx, 0 <= m < 80
    int times;  //ダイナミクスの回数
};

amNet::amNet(){
    std::cout << "Associative Memory Net" << std::endl;
    xa.resize(M, std::vector<double>(N));
    w.resize(N, std::vector<double>(N));

    std::random_device rnd;     // 非決定的な乱数生成器を生成, /dev/randomとかを見たりする. シード値の代わりに使う．
    std::mt19937 mt(rnd());     // mersenne twister 32bit 擬似乱数生成器
    std::uniform_int_distribution<> rand2(0, 1);        // [0, 1] 範囲の一様乱数を生成
    int rand2arr[] = {-1, 1};

    std::cout << "Set random memory." << std::endl;
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            xa.at(i).at(j) = rand2arr[rand2(mt)];
        }
    }

    std::cout << "Set weight." << std::endl;
    for(int i=0; i<N; i++){
        for(int j=0; j<=i; j++){
            double sum = 0.0;
            for(int a=0; a<M; a++){
                sum += xa.at(a).at(i) * xa.at(a).at(j);
            }
            w.at(i).at(j) = sum / (double)N;
            w.at(j).at(i) = w.at(i).at(j);
        }
    }

    for(int i=0; i<N; i++){
        w.at(i).at(i) = 0.0;
    }
}

double amNet::sgn(std::vector<double>& bx, int i){
    double sum = 0.0;
    for(int j=0; j<N; j++){
        sum += w.at(i).at(j) * bx.at(j);
    }
    if(sum > 0.0){
        return 1.0;
    }else{
        return -1.0;
    }
}

void amNet::flip_set_x(int m, int a){
    this->m = m;
    x.resize(N);
    for(int i=0; i<N; i++){
        if(i < a){
            x.at(i) = xa.at(m).at(i) * -1;
        }else{
            x.at(i) = xa.at(m).at(i);
        }
    }
}

void amNet::run(int times, int flip_xa, std::vector<int>& flip_list){
    this->times = times;
    this->m = flip_xa - 1;
    for(int i=0; i<flip_list.size(); i++){
        std::cout << "flip a =  " << flip_list.at(i) << std::endl;
        flip_set_x(m, flip_list.at(i));
        update();
    }
}

void amNet::update(){
    std::vector<double> res(times + 1);
    res.at(0) = directon_cos();         // NOTE:0回目をプロットする，しなかったら大きい変化が見えなくなる．
    for(int t=1; t<=times; t++){
        std::vector<double> bx(x);
        for(int i=0; i<N; i++){
            x.at(i) = sgn(bx, i);
        }
        res.at(t) = directon_cos();
    }
    result.push_back(res);
}

double amNet::directon_cos(){
    double sum = 0.0;
    for(int i=0; i<N; i++){
        sum += xa.at(m).at(i) * x.at(i);
    }
    return sum / (double)N;
}

void amNet::out_result(){
    FILE *out_file;
    out_file = fopen("am.csv", "w");

    for(int i=0; i<=times; i++){
        fprintf(out_file, "%d,", i);
        for (int j = 0; j < result.size(); j++){
            fprintf(out_file, "%8.7lf,", result.at(j).at(i));
        }
        fprintf(out_file, "\n");
    }
    fclose(out_file);
}

int main(){
    amNet net;
    std::vector<int> flip_list;
    for(int i=0; i<N/2; i+=25){
        flip_list.push_back(i);
    }
    
    net.run(20, 1, flip_list);
    net.out_result();
    return 0;
}