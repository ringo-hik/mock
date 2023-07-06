#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "../Project1/cal.cpp"

using ::testing::ElementsAre;

class CalMock : public cal
{
    MOCK_METHOD(int, getSum, (int a, int b), ());
};
TEST(CheckValueTest, TestPass) {
    CalMock mock;
}

TEST(CheckValueTest, TestFail) {
    int value = 20;
    EXPECT_EQ(CheckValue(value), "fail");

    std::vector<int> expected_values = { 20, 21, 22, 23, 24, 25 };
    EXPECT_THAT(expected_values, ElementsAre(20, 21, 22, 23, 24, 25));
}