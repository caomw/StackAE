#include "StackAutoEncoder.h"

int main()
{
    StackAutoEncoder SSAE(1, 50000, 784, 10, 10000);
    
    SSAE.FeedForward();
    SSAE.Test();
}