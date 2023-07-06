#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "../Project1/cal.cpp"

using namespace testing;

class CalMock {
public:
    MOCK_METHOD(int, add, (int a, int b));
    MOCK_METHOD(int, subtract, (int a, int b));
    MOCK_METHOD(int, multiply, (int a, int b));
};

TEST(CalTest, Add) {
    CalMock mock;

    EXPECT_CALL(mock, add(2, 3))
        .WillOnce(Return(7));

    int result = mock.add(2, 3);

    EXPECT_THAT(result, Eq(5));
}

TEST(CalTest, Subtract) {
    CalMock mock;

    EXPECT_CALL(mock, subtract(5, 2))
        .WillOnce(Return(3));

    int result = mock.subtract(5, 2);

    EXPECT_THAT(result, Eq(3));
}

TEST(CalTest, Multiply) {
    CalMock mock;

    EXPECT_CALL(mock, multiply(4, 3))
        .WillOnce(Return(12));

    int result = mock.multiply(4, 3);

    EXPECT_THAT(result, Eq(12));
}