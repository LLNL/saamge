/*! \file
    \brief Configuration manager.

    SAAMGE: smoothed aggregation element based algebraic multigrid hierarchies
            and solvers.

    Copyright (c) 2016, Lawrence Livermore National Security,
    LLC. Developed under the auspices of the U.S. Department of Energy by
    Lawrence Livermore National Laboratory under Contract
    No. DE-AC52-07NA27344. Written by Delyan Kalchev, Andrew T. Barker,
    and Panayot S. Vassilevski. Released under LLNL-CODE-667453.

    This file is part of SAAMGE. 

    Please also read the full notice of copyright and license in the file
    LICENSE.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License (as
    published by the Free Software Foundation) version 2.1 dated February
    1999.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
    conditions of the GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this program; if not, see
    <http://www.gnu.org/licenses/>.

    XXX: Notes

    In distributed parallelism case, all processes will have a copy of all
    configuration options. If one process changes a value of a configuration
    option and all others do not do the same, then this most likely would be an
    issue since the behaviour of different processes will not be the same. This
    module does not provide any means of synchronizing options' values. At this
    point this is not considered necessary.
*/

#pragma once
#ifndef _CONFIG_MGR_HPP
#define _CONFIG_MGR_HPP

/* Defines */

/*! \brief Begins a declaration block of a configuration class.

    This starts the declaration block of a class. Inside the block
    \b CONFIG_DECLARE_OPTION is used and the block is closed by
    \b CONFIG_END_CLASS_DECLARATION.

    Typically in a header (.hpp) file you have something like (pay attention to
    the semicolons):

        CONFIG_BEGIN_CLASS_DECLARATION(DIMENSIONS)

            CONFIG_DECLARE_OPTION(double, width);
            CONFIG_DECLARE_OPTION(double, length);
            CONFIG_DECLARE_OPTION(SparseMatrix *, ptr);

        CONFIG_END_CLASS_DECLARATION(DIMENSIONS)

    \param class_name (IN) The name of the class. Usual C and C++ naming
                           requirements apply.
*/
#define CONFIG_BEGIN_CLASS_DECLARATION(class_name) \
    struct __config_class_##class_name##_ { \
        __config_class_##class_name##_();

/*! \brief Closes the declaration block of a configuration class.

    See \b CONFIG_BEGIN_CLASS_DECLARATION.

    \param class_name (IN) The name of the class. Usual C and C++ naming
                           requirements apply.
*/
#define CONFIG_END_CLASS_DECLARATION(class_name) \
    }; \
    extern struct __config_class_##class_name##_ \
                  __config_class_##class_name##_instance_;

/*! \brief Adds (declares) an option in a configuration class.

    See \b CONFIG_BEGIN_CLASS_DECLARATION.

    Every option declaration is followed by a semicolon (;).

    \param option_type (IN) The type of the option. These are types usually
                            used for declaring variables or objects.
    \param option_name (IN) The name of the option. Usual C and C++ naming
                            requirements apply.
*/
#define CONFIG_DECLARE_OPTION(option_type, option_name) \
    option_type const __default_##option_name##_value_; \
    option_type       __current_##option_name##_value_

/*! \brief Begins a block of default values for a configuration class.

    This is a version of \b CONFIG_BEGIN_INLINE_CLASS_DEFAULTS to be used in a
    source (.cpp) file. It is used the same way.

    \param class_name (IN) The name of the class. Usual C and C++ naming
                           requirements apply.
*/
#define CONFIG_BEGIN_CLASS_DEFAULTS(class_name) \
    __config_class_##class_name##_::__config_class_##class_name##_() :

/*! \brief Begins a block of default values for a configuration class.

    AFTER declaring the class with all its options (see
    \b CONFIG_BEGIN_CLASS_DECLARATION) the default values of the options are
    set.

    This is typically used in a header (.hpp) file. You have something like
    (pay attention to the comma separators):

        CONFIG_BEGIN_INLINE_CLASS_DEFAULTS(DIMENSIONS)

            CONFIG_DEFINE_OPTION_DEFAULT(width, 5.3),
            CONFIG_DEFINE_OPTION_DEFAULT(length, 10.7),
            CONFIG_DEFINE_OPTION_DEFAULT(ptr, NULL)

        CONFIG_END_CLASS_DEFAULTS

    \param class_name (IN) The name of the class. Usual C and C++ naming
                           requirements apply.

    \warning Have in mind that even if you modify any default value in a header
             (.hpp) file, it will only take effect when the source (.cpp) file
             containing the respective \b CONFIG_DEFINE_CLASS is recompiled and
             linked to the final executable.
*/
#define CONFIG_BEGIN_INLINE_CLASS_DEFAULTS(class_name) \
    inline CONFIG_BEGIN_CLASS_DEFAULTS(class_name)

/*! \brief Closes a block of default values for a configuration class.

    See \b CONFIG_BEGIN_INLINE_CLASS_DEFAULTS.

    It closes a block started either by \b CONFIG_BEGIN_INLINE_CLASS_DEFAULTS
    or by \b CONFIG_BEGIN_CLASS_DEFAULTS.
*/
#define CONFIG_END_CLASS_DEFAULTS \
    {}

/*! \brief Defines the default value of an option in a configuration class.

    See \b CONFIG_BEGIN_INLINE_CLASS_DEFAULTS.

    The default values' definitions are separated by commas (,).

    \param option_name (IN) The name of the option. Usual C and C++ naming
                            requirements apply.
    \param option_default_value (IN) The default value for the option.
*/
#define CONFIG_DEFINE_OPTION_DEFAULT(option_name, option_default_value) \
    __default_##option_name##_value_(option_default_value), \
    __current_##option_name##_value_(option_default_value)

/*! \brief Defines a configuration class.

    Defines in the sense that the class itself acquires memory to save its
    options' values. It is called in a source file after declaring the class
    (see \b CONFIG_BEGIN_INLINE_CLASS_DEFAULTS) and defining its options'
    default values (see \b CONFIG_BEGIN_INLINE_CLASS_DEFAULTS).

    E.g., in a source (.cpp) file you have something like:

        CONFIG_DEFINE_CLASS(DIMENSIONS);

    \param class_name (IN) The name of the class. Usual C and C++ naming
                           requirements apply.
*/
#define CONFIG_DEFINE_CLASS(class_name) \
    struct __config_class_##class_name##_ \
           __config_class_##class_name##_instance_

/*! \brief Provides access to an option in a configuration class.

    This macro provides access to the current value of an option in a
    configuration class as a modifiable lvalue. E.g., you can get the option
    value as a right-hand-side value in an assignment like:

        double foo = CONFIG_ACCESS_OPTION(DIMENSIONS, width);

    Also, you can set the value to an option by using it as a left-hand side of
    an assignment like:

        CONFIG_ACCESS_OPTION(DIMENSIONS, width) = 15.9;

    Finally, it can be used as a function argument passed by reference which
    allows modifying the option's value and/or setting/reading options of
    compound types (like structures).

    \param class_name (IN) The name of the class. Usual C and C++ naming
                           requirements apply.
    \param option_name (IN) The name of the option. Usual C and C++ naming
                            requirements apply.

    \returns See the description.
*/
#define CONFIG_ACCESS_OPTION(class_name, option_name) \
    __config_class_##class_name##_instance_.__current_##option_name##_value_

/*! \brief Returns the immutable default value of an option.

    This is not the current value of the option but rather its default value
    set at the beginning (see \b CONFIG_BEGIN_INLINE_CLASS_DEFAULTS).

    \param class_name (IN) The name of the class. Usual C and C++ naming
                           requirements apply.
    \param option_name (IN) The name of the option. Usual C and C++ naming
                            requirements apply.

    \returns The preset default value of the desired option in the desired
             configuration class.
*/
#define CONFIG_VIEW_OPTION_DEFAULT(class_name, option_name) \
    __config_class_##class_name##_instance_.__default_##option_name##_value_

/*! \brief Resets an option to its default value.

    Additionally, it can be used as a righ-hand-side value giving the default
    value of an option. E.g., the code:

        double foo = CONFIG_RESET_OPTION(DIMENSIONS, width);

    will reset the option \em width in the configuration class \em DIMENSIONS
    to its default value and will assign this default value (which in our case
    is the one given in the example for \b CONFIG_BEGIN_INLINE_CLASS_DEFAULTS)
    to the variable \em foo.

    \param class_name (IN) The name of the class. Usual C and C++ naming
                           requirements apply.
    \param option_name (IN) The name of the option. Usual C and C++ naming
                            requirements apply.

    \returns The preset default value of the desired option in the desired
             configuration class.
*/
#define CONFIG_RESET_OPTION(class_name, option_name) \
   (__config_class_##class_name##_instance_.__current_##option_name##_value_ = \
    __config_class_##class_name##_instance_.__default_##option_name##_value_)

#endif // _CONFIG_MGR_HPP
