#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX 100

struct Node {
    char data;
    struct Node *left, *right;
};

struct Node *nodeStack[MAX];
char operatorStack[MAX];
int ntop = -1, otop = -1;

struct Node *createNode(char ch)
{
    struct Node *newNode =
        (struct Node *)malloc(sizeof(struct Node));

    newNode->data = ch;
    newNode->left = newNode->right = NULL;

    return newNode;
}

int precedence(char op)
{
    if (op == '+' || op == '-')
        return 1;

    if (op == '*' || op == '/')
        return 2;

    return 0;
}

void buildSubtree()
{
    struct Node *root = createNode(operatorStack[otop--]);

    root->right = nodeStack[ntop--];
    root->left  = nodeStack[ntop--];

    nodeStack[++ntop] = root;
}

struct Node *constructTree(char exp[])
{
    int i;
    char ch;

    for (i = 0; exp[i] != '\0'; i++) {

        ch = exp[i];

        if (isalnum(ch)) {
            nodeStack[++ntop] = createNode(ch);
        }

        else if (ch == '(') {
            operatorStack[++otop] = ch;
        }

        else if (ch == ')') {

            while (operatorStack[otop] != '(')
                buildSubtree();

            otop--;   // remove '('
        }

        else {
            while (otop != -1 &&
                   operatorStack[otop] != '(' &&
                   precedence(operatorStack[otop]) >= precedence(ch)) {

                buildSubtree();
            }

            operatorStack[++otop] = ch;
        }
    }

    while (otop != -1)
        buildSubtree();

    return nodeStack[ntop--];
}

void prefix(struct Node *root)
{
    if (root != NULL) {
        printf("%c", root->data);
        prefix(root->left);
        prefix(root->right);
    }
}

void postfix(struct Node *root)
{
    if (root != NULL) {
        postfix(root->left);
        postfix(root->right);
        printf("%c", root->data);
    }
}

int main()
{
    char exp[MAX];
    struct Node *root;

    printf("Enter expression: ");
    scanf("%s", exp);

    root = constructTree(exp);

    printf("\nPrefix  : ");
    prefix(root);

    printf("\nPostfix : ");
    postfix(root);

    printf("\n");

    return 0;
}
