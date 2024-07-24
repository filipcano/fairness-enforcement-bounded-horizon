#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<chrono>
#include<cmath>
#include<cstdlib>
#include<algorithm>
#include<cassert>
#include<random>
// #include "boost/multi_array.hpp"
using namespace std;

// MACROS //
const int GROUP_A = 0;
const int GROUP_B = 1;
const int ACCEPT = 1;
const int REJECT = 0;
const float INF = std::numeric_limits<float>::infinity();

typedef vector<int> VI;
typedef vector<float> VD;
typedef vector<VD> V2D;
typedef vector<V2D> V3D;
typedef vector<V3D> V4D;
typedef vector<V4D> V5D;
// typedef vector<V5D> V6D;
// typedef vector<V6D> V7D;
// typedef vector<V7D> V8D;
// typedef vector<V8D> V9D;



float custom_multiply(float a, float b) {
    //multiply two floats, with inf*0 = inf
    if (std::isinf(a) && b == 0.0f) {
        return std::numeric_limits<float>::infinity();
    } else if (std::isinf(b) && a == 0.0f) {
        return std::numeric_limits<float>::infinity();
    } else {
        float result = a * b;
        if (a != 0.0f && b != 0.0f && result == 0.0f) {
            // Return the smallest positive normal float
            return std::numeric_limits<float>::min();
        }
        return result;
    }
}

int sampleNumber(const std::vector<float>& probabilities) {
    // returns an int between 0 and probabilities.size()-1

    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());

    return dist(gen);
}

void PrintMemoryInfo(string extra) {
    std::string line;
    std::ifstream statusFile("/proc/self/status");

    while (getline(statusFile, line)) {
        if (line.substr(0, 6) == "VmSize") {
            std::string memStr = line.substr(7);
            memStr = memStr.substr(0, memStr.find(" kB"));
            int memKb = std::stoi(memStr);
            float memMb = memKb / 1024.0;
            std::cout << extra << " Memory Usage: " << memMb << " MB" << std::endl;
            break;
        }
    }
}

VI convertToBaseB(int N, int B) {
    VI digits(0);

    while (N > 0) {
        digits.push_back(N % B);
        N /= B;
    }

    // The digits are stored in reverse order, so we need to reverse them back
    reverse(digits.begin(), digits.end());
    return digits;
}



class DPEnforcerMinCost {
    V4D VAL;
    // VAL[remaining_decisions][Group A accepted][Group B accepted][Group A seen]
    // VAL[remaining_decisions][Group A seen][Group A accepted][Group A acc]
    // remaining_decisions: 0 ... T
    // Group A seen: 0 ... T-remaining_decisions
    // Group A accepted: 0 ... gAseen //
    // Group B accepted : 0 ... T-remaining_decisions-gAseen
    // Group B observed is taken from Group A seen, T and remaining decisions, as gBseen = T-remaining_decisions-gAseen

    // Deprecated: 
    // VAL[remaining_decisions][Group A accepted][Group B accepted][Group A seen]
    // remaining_decisions: 0 ... T
    // Group A accepted: 0 ... T
    // Group B accepted: 0 ... T // this could be optimizied, T is too bias, would not pass the constraint
    // Group A observed: 0 ... T
    // Group B observed is taken from Group A, T and remaining decisions



    VD get_threshold(int t, int gAacc, int gBacc, int gAseen);
    
    
public:
    DPEnforcerMinCost(bool dynamic_distribution);
    int T; // horizon length
    int X; // total number of costs
    VD N; // sequence of costs, of size X
    float eps; // threshold
    float alpha, beta; // min and max acceptance rates (for composability)
    int buff_gAacc, buff_gAseen, buff_gBacc, buff_gBseen; // buffer values to compute buffered DP
    int defaultVal;

    V4D Prob; // Prob[i][j][k][l] prob of sampling group j, and decision k and value l at timestep i

    
    float Val(int t, int gAseen, int gAacc, int gBacc);
    float compute_dp(int gAseen, int gAacc, int gBseen, int gBacc);
    void printInputs();
    V2D make_one_simulation();
    void save_val_to_file(string filename);
    void load_val_from_file(string filename);
};

DPEnforcerMinCost::DPEnforcerMinCost(bool dynamic_distribution) {
    cin >> T >> X >> eps >> alpha >> beta;
    cin >> buff_gAacc >> buff_gAseen >> buff_gBacc >> buff_gBseen;

    N = VD(X,0);
    int minval = 1e9;
    int maxval = -minval;
    for (int i = 0; i < X; ++i) {
        cin >> N[i];
        minval = min((float)minval, N[i]);
        maxval = max((float)maxval, N[i]);
    }
    defaultVal = min(minval*X-1, -1);
    Prob = V4D(T+1, V3D(2,  V2D(2, VD(X, 0))));

    int n_prob_distributions = dynamic_distribution ? T+1 : 1;

    for (int i=0; i < n_prob_distributions; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 1; k >= 0; --k) {
                for (int l = 0; l < X; ++l)
                    cin >> Prob[i][j][k][l];
            }
        }
    }

    if (!dynamic_distribution) {
        for (int i=1; i < T+1; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 1; k >= 0; --k) {
                    for (int l = 0; l < X; ++l)
                        Prob[i][j][k][l] = Prob[0][j][k][l];
                }
            }
        }  
    }


    
    // VAL = V4D(T+1, V3D(T+1, V2D(T+1, VD(T+1, defaultVal))));

    VAL = V4D(T+1);
    for (int rem_dec = 0; rem_dec <= T; ++rem_dec) {
        int t = T - rem_dec;
        VAL[rem_dec] = V3D(t+1);
        for (int gAseen = 0; gAseen <= t; ++gAseen) {
            VAL[rem_dec][gAseen] = V2D(gAseen+1);
            for (int gAacc = 0; gAacc <= gAseen; ++gAacc) {
                VAL[rem_dec][gAseen][gAacc] = VD(t-gAseen+1, defaultVal);
            }
        }
    }
}


void DPEnforcerMinCost::printInputs() {
    cout << "Start print inputs" << endl;
    cout << "T: " << T << ", X: " << X << ", eps: " << eps << endl << "N: [";
    for (int i = 0; i < N.size(); ++i) cout << N[i] << ", ";
    cout << endl << "Probs: \n";
    for (int t = 0; t < Prob.size(); ++t) {
        cout << "t=" << t << ": ";
        for (int group = 0; group < Prob[0].size();++group) {
            for (int decision = 0; decision < Prob[0][0].size(); ++decision) {
                for (int k = 0; k < Prob[0][0][0].size(); ++k) {
                    cout << Prob[t][group][decision][k] << ", ";
                }
            }
        }
        cout << endl;
    }
    cout << "Finished print inputs" << endl;

}


void DPEnforcerMinCost::save_val_to_file(string filename) {
    // Serialize
    cout << "Saving policy in " << filename << " ..." << endl;

    std::ofstream file(filename);
    if (file.is_open()) {
        for (int i1 =0; i1 < VAL.size(); ++i1) {
            for (int i2=0; i2 < VAL[i1].size(); ++i2) {
                for (int i3=0; i3< VAL[i1][i2].size(); ++i3) {
                    for (int i4=0; i4< VAL[i1][i2][i3].size(); ++i4) {
                        float val = VAL[i1][i2][i3][i4];
                        bool print = true;
                        if (val == 0.0) {
                            print = false;
                        }
                        else if (val == INF) {
                            int gAseen = i2;
                            int gAacc = i3;
                            int gBacc = i4;
                            int gBseen = T - i1 - gAseen;
                            float dp = abs((float)gAacc/(1.0+gAseen) - (float)gBacc/(1.0+gBseen));
                            if (dp < eps) print = true;
                            else print = false;
                        }
                        else if (val == defaultVal) print = false;
                        
                        if (print) {
                            file << i1 << " " << i2 << " " << i3 << " " << i4;
                            file << " " << VAL[i1][i2][i3][i4] << "\n";
                        }
                    }
                }
            }
        }
        file.close();
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
}

void DPEnforcerMinCost::load_val_from_file(string filename) {
    cout << "Loading policy..." << endl;
    std::ifstream file(filename);
    int i1, i2, i3, i4;
    if (file.is_open()) {
        while (file >> i1 >> i2 >> i3 >> i4) {
            file >> VAL[i1][i2][i3][i4];
        }
        file.close();
    } else {
        std::cerr << "Unable to open file for reading." << std::endl;
    }
}




VD DPEnforcerMinCost::get_threshold(int t, int gAacc, int gBacc, int gAseen) {
    VD threshold(2,0);
    float acceptA = Val(t-1, gAseen+1, gAacc +1, gBacc);
    float rejectA = Val(t-1, gAseen+1, gAacc, gBacc);
    float acceptB = Val(t-1, gAseen, gAacc, gBacc+1);
    float rejectB = Val(t-1, gAseen, gAacc, gBacc);

    int min_it = 0;
    while ((min_it < N.size()) and (rejectA > acceptA + N[min_it])) ++min_it;
    if ((rejectA == acceptA + N[min_it]) or (min_it == N.size()-1)) {
        threshold[GROUP_A] =  N[min_it];
    }
    else if (min_it == 0) {
        threshold[GROUP_A] = -INF;
    }
    else {
        threshold[GROUP_A] = ((float)N[min_it] + (float)N[min_it +1])/2;
    }

    min_it = 0;
    while ((min_it < N.size()) and (rejectB > acceptB + N[min_it])) ++min_it;
    if ((rejectB == acceptB + N[min_it]) or (min_it == N.size()-1)) {
        threshold[GROUP_B] =  N[min_it];
    }
    else if (min_it == 0) {
        threshold[GROUP_B] = -INF;
    }
    else {
        threshold[GROUP_B] = ((float)N[min_it] + (float)N[min_it +1])/2;
    }
    return threshold;
}

float DPEnforcerMinCost::compute_dp(int gAseen, int gAacc, int gBseen, int gBacc) {
    int a = gAacc + buff_gAacc;
    int b = gAseen + buff_gAseen;
    int p = gBacc + buff_gBacc;
    int q = gBseen + buff_gBseen;

    if (b + q == 0) return 0;
    return abs((float)a/(1.0 + b) - (float)p/(1.0 + q));
}

float DPEnforcerMinCost::Val(int t, int gAseen, int gAacc, int gBacc) {
    // cout << "hola1" << endl;
    // cout << "t: " << t << ", gAacc: " <<  gAacc << ", gBacc: " << gBacc << ", gAseen: " << gAseen << endl;
    float& res = VAL[t][gAseen][gAacc][gBacc];
    // cout << "t: " << t << ", gAacc: " <<  gAacc << ", gBacc: " << gBacc << endl << endl;
    // cout << ", gAseen: " << gAseen << ", gBseen" << (X-t)-gAseen << ", res: " << res << endl; 
    if (res != defaultVal) return res;
    // cout << "t: " << t << ", gAacc: " <<  gAacc << ", gBacc: " << gBacc << ", gAseen: " << gAseen << endl;
    int gBseen = (T-t)-gAseen;
    if (t == 0) {
        float bias = compute_dp(gAseen, gAacc, gBseen, gBacc);
        // bool accRateA = (alpha*gAseen <= gAacc) and (gAacc <= beta*gAseen);
        bool accRateA = (custom_multiply(alpha,gAseen) <= gAacc) and (gAacc <= custom_multiply(beta,gAseen));
        bool accRateB = (custom_multiply(alpha,gBseen) <= gBacc) and (gBacc <= custom_multiply(beta,gBseen));

        if ((bias < eps) and accRateA and accRateB) {
            return res = 0;
        } 
        return res = INF;
    }

    res = 0;

    for (int group = 0; group < 2; ++group) {
        for (int k = 0; k < X; ++k) {
            for (int decision = 0; decision < 2; ++decision) {
                float accept = 0;
                float reject = 0;
                if (group == GROUP_A and decision == ACCEPT) {
                    accept = Val(t-1,gAseen+1, gAacc +1,gBacc);
                    reject = N[k] + Val(t-1, gAseen+1, gAacc, gBacc);
                }
                else if (group == GROUP_A and decision == REJECT) {
                    accept = N[k] + Val(t-1, gAseen+1, gAacc +1,gBacc);
                    reject = Val(t-1,gAseen+1, gAacc, gBacc);
                }
                else if (group == GROUP_B and decision == ACCEPT)  { 
                    accept = Val(t-1,gAseen, gAacc, gBacc+1);
                    reject = N[k] + Val(t-1, gAseen, gAacc, gBacc);
                }
                else { // group == GROUP_B and decision == REJECT
                    accept = N[k] + Val(t-1, gAseen, gAacc, gBacc+1);
                    reject = Val(t-1, gAseen, gAacc, gBacc);
                }
                res += custom_multiply(Prob[t][group][decision][k],min(accept, reject));
            }
        }
    }
    return res;
}


int main(int argc, char **argv) {
    bool save_policy = false;
    bool load_policy = false;
    bool dynamic_distribution = false;
    int n_simulations = 1;
    string saved_policy_file = "default";
    for (int i = 0 ; i < argc; ++i) {
        string argument = argv[i];
        if (argument == "--save_policy") save_policy = true;
        if (argument == "--load_policy") load_policy = true;
        if (argument == "--dynamic_distribution") dynamic_distribution = true;
        string keyword = "--n_simulations=";
        if (argument.find(keyword) != std::string::npos) 
            n_simulations = stoi(argument.substr(keyword.size(),argument.size()-keyword.size()));
        keyword = "--saved_policy_file=";
        if (argument.find(keyword) != std::string::npos) 
            saved_policy_file = argument.substr(keyword.size(),argument.size()-keyword.size());
    }
    DPEnforcerMinCost FA = DPEnforcerMinCost(dynamic_distribution);
    // FA.printInputs();
    cout << "T: " << FA.T << endl;
    string filename = "saved_policies/" + to_string(FA.T) + "_" + to_string(FA.X) + "_agent.txt";
    if (saved_policy_file != "default") filename = saved_policy_file;
    if (load_policy) FA.load_val_from_file(filename);

    auto t_start = std::chrono::high_resolution_clock::now();
    
    float aux = FA.Val(FA.T, 0, 0, 0);
    cout << "Value: " << aux << endl;
    auto t_end = std::chrono::high_resolution_clock::now();
    float elapsed_time_ms = std::chrono::duration<float, std::milli>(t_end-t_start).count();
    cout << "Optimized Elapsed time: " << elapsed_time_ms/1000 << " seconds" << endl;
    PrintMemoryInfo("Optimized");
    if (save_policy) FA.save_val_to_file(filename);
    

}