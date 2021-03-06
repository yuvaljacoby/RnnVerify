/*********************                                                        */
/*! \file Equation.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz, Shantanu Thakoor
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

 **/

#include "Equation.h"
#include "FloatUtils.h"
#include "Map.h"

Equation::Addend::Addend( double coefficient, unsigned variable )
    : _coefficient( coefficient )
    , _variable( variable )
{
}

Equation::Addend::Addend( const Addend& add )
    : _coefficient( add._coefficient )
    , _variable( add._variable )
{
}

bool Equation::Addend::operator==( const Addend &other ) const
{
    return ( _coefficient == other._coefficient ) && ( _variable == other._variable );
}

double Equation::Addend::getCoefficient()
{
    return _coefficient;
}

unsigned Equation::Addend::getVariable(){
    return _variable;
}

Equation::Equation()
    : _type( Equation::EQ )
{
}

Equation::Equation( EquationType type )
    : _type( type )
{
}

Equation::Equation(const Equation& eq)
    : _type(eq._type)
    {
    _type = eq._type;
    _scalar = eq._scalar;
    for ( const auto &addend : eq._addends )
        _addends.append( Addend( addend ));
    /*_addends == eq._addends;*/
}

void Equation::addAddend( double coefficient, unsigned variable )
{
    _addends.append( Addend( coefficient, variable ) );
}

std::vector<Equation::Addend> Equation::getAddends() {
    // TODO: Find a way to return List in pybind11 and than we change here to return List instead of the copy
    return { std::begin(_addends), std::end(_addends) };

//    return v;
}

void Equation::setScalar( double scalar )
{
    _scalar = scalar;
}

Equation::EquationType Equation::getType()
{
    return _type;
}

void Equation::setType( EquationType type )
{
    _type = type;
}

double Equation::getScalar()
{
    return _scalar;
}

void Equation::updateVariableIndex( unsigned oldVar, unsigned newVar )
{
    // Find oldVar's addend and update it
    List<Addend>::iterator oldVarIt = _addends.begin();
    while ( oldVarIt != _addends.end() && oldVarIt->_variable != oldVar )
        ++oldVarIt;

    // OldVar doesn't exist - can stop
    if ( oldVarIt == _addends.end() )
        return;

    // Update oldVar's index
    oldVarIt->_variable = newVar;

    // Check to see if there are now two addends for newVar. If so,
    // remove one and adjust the coefficient
    List<Addend>::iterator newVarIt;
    for ( newVarIt = _addends.begin(); newVarIt != _addends.end(); ++newVarIt )
    {
        if ( newVarIt == oldVarIt )
            continue;

        if ( newVarIt->_variable == newVar )
        {
            oldVarIt->_coefficient += newVarIt->_coefficient;
            _addends.erase( newVarIt );
            return;
        }
    }
}

bool Equation::operator==( const Equation &other ) const
{
    return
        ( _addends == other._addends ) &&
        ( _scalar == other._scalar ) &&
        ( _type == other._type );
}

bool Equation::equivalent( const Equation &other ) const
{
    if ( _scalar != other._scalar )
        return false;

    if ( _type != other._type )
        return false;

    Map<unsigned, double> us;
    Map<unsigned, double> them;

    for ( const auto &addend : _addends )
        us[addend._variable] = addend._coefficient;

    for ( const auto &addend : other._addends )
        them[addend._variable] = addend._coefficient;

    return us == them;
}

void Equation::dump() const
{
    for ( const auto &addend : _addends )
    {
        if ( FloatUtils::isZero( addend._coefficient ) )
            continue;

        if ( FloatUtils::isPositive( addend._coefficient ) )
            printf( "+" );

        printf( "%.6lfx%u ", addend._coefficient, addend._variable );
    }

    switch ( _type )
    {
    case Equation::GE:
        printf( " >= " );
        break;

    case Equation::LE:
        printf( " <= " );
        break;

    case Equation::EQ:
        printf( " = " );
        break;
    }

    printf( "%.6lf\n", _scalar );
}

bool Equation::isVariableMergingEquation( unsigned &x1, unsigned &x2 ) const
{
    if ( _addends.size() != 2 )
        return false;

    if ( _type != Equation::EQ )
        return false;

    if ( !FloatUtils::isZero( _scalar ) )
        return false;

    double coefficientOne = _addends.front()._coefficient;
    double coefficientTwo = _addends.back()._coefficient;

    if ( FloatUtils::isZero( coefficientOne ) || FloatUtils::isZero( coefficientTwo ) )
        return false;

    if ( FloatUtils::areEqual( coefficientOne, -coefficientTwo ) )
    {
        x1 = _addends.front()._variable;
        x2 = _addends.back()._variable;
        return true;
    }

    return false;
}

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
