##!python
## -*- coding: utf-8 -*-

"""
This Python module calculates the Credit Value Adjustment for a single netting set of plain vanilla
interest rate swaps.

The code is based on the IPython Notebook of Matthias Groncki (see reference below).

References:

"CVA Calculation with QuantLib and Python", Matthias Groncki
    - https://ipythonquant.wordpress.com/tag/cva/
    - http://nbviewer.ipython.org/github/mgroncki/IPythonScripts/blob/master/CVA_calculation_I.ipynb

"FOOLING AROUND WITH QUANTLIB: GSR MODEL", Peter Caspers:
    - https://quantlib.wordpress.com/tag/gsr-model/

"One Factor Gaussian Short Rate Model Implementation", Peter Caspers, March 1, 2013:
    - http://papers.ssrn.com/sol3/papers.cfm?abstract_id=2246013

"""

# TODO PyTorch tensors, data sets, data loader and samplers, deep modeling with torch.nn...


import numpy as np
import torch
# import matplotlib.pyplot as plt

# Check version of QuantLib...
import QuantLib as ql


def get_version():
    return list(map(int, ql.__version__.split('.')))


if get_version()[1] < 6:
    print('You need QuantLib version 1.6 or higher!')
    exit()

# General QuantLib functions...
set_evaluation_date = lambda date: ql.Settings.instance().setEvaluationDate(date)

link_to_curve = lambda relinkable_handle, curve: relinkable_handle.linkTo(curve)


# Random numbers...
def create_random_number_generator(evaluation_time_grid, seed=1):

    uniform_rng = ql.MersenneTwisterUniformRng(seed)
    uniform_rsg = ql.MersenneTwisterUniformRsg(len(evaluation_time_grid) - 1, uniform_rng)

    return ql.InvCumulativeMersenneTwisterGaussianRsg(uniform_rsg)


# Default curve...
def create_default_curve(default_dates, hazard_rates, day_count=ql.Actual365Fixed()):

    default_curve = ql.HazardRateCurve(default_dates, hazard_rates, day_count)
    default_curve.enableExtrapolation()

    return default_curve


def get_default_probability(times, default_curve):

    # TODO ...
    # torch.FloatTensor(default_curve.defaultProbability)

    return np.vectorize(default_curve.defaultProbability)(times)


def get_survival_probability(times, default_curve):
    return np.vectorize(default_curve.survivalProbability)(times)


def get_default_density(times, default_curve):
    return np.vectorize(default_curve.defaultDensity)(times)


def get_hazard_rate(times, default_curve):
    """

    @param times:
    @param default_curve:
    @return:
    """

    return np.vectorize(default_curve.hazardRate)(times)


def calculate_default_probability_grid(evaluation_time_grid, default_curve):
    return np.vectorize(default_curve.defaultProbability)(
        evaluation_time_grid[:-1], evaluation_time_grid[1:]
    )


# Discount curve...
def create_flat_forward(todays_date, rate, day_count=ql.Actual365Fixed()):

    flat_forward = ql.FlatForward(todays_date, ql.QuoteHandle(rate), day_count)
    flat_forward.enableExtrapolation()

    return flat_forward, \
           ql.YieldTermStructureHandle(flat_forward), \
           ql.RelinkableYieldTermStructureHandle(flat_forward)


def generate_discount_factors(flat_forward_handle, evaluation_time_grid):
    return np.vectorize(flat_forward_handle.discount)(evaluation_time_grid)


def get_discount_curve(curve_dates,
                       discount_factors,
                       day_count_convention=ql.Actual365Fixed()):

    discount_curve = ql.DiscountCurve(
        curve_dates,
        discount_factors,
        day_count_convention
    )

    discount_curve.enableExtrapolation()

    return discount_curve


# Pricing engine...
def create_pricing_engine(flat_forward_relinkable_handle):
    return ql.DiscountingSwapEngine(flat_forward_relinkable_handle)


# Swap portfolio...
def create_plain_vanilla_swap(start_date, maturity_date,
                              nominal_amount,
                              float_index,
                              fixed_rate,
                              fixed_leg_tenor=ql.Period("1y"),
                              fixed_leg_business_day_convention=ql.ModifiedFollowing,
                              fixed_leg_day_count_convention=ql.Thirty360(ql.Thirty360.BondBasis),
                              calendar=ql.Sweden(),
                              spread=0.0,
                              swap_type=ql.VanillaSwap.Payer):

    end_date = calendar.advance(start_date, maturity_date)

    fixed_schedule = ql.Schedule(
        start_date,
        end_date,
        fixed_leg_tenor,
        float_index.fixingCalendar(),
        fixed_leg_business_day_convention,
        fixed_leg_business_day_convention,
        ql.DateGeneration.Backward,
        False
    )

    float_schedule = ql.Schedule(
        start_date,
        end_date,
        float_index.tenor(),
        float_index.fixingCalendar(),
        float_index.businessDayConvention(),
        float_index.businessDayConvention(),
        ql.DateGeneration.Backward,
        False
    )

    swap = ql.VanillaSwap(
        swap_type,
        nominal_amount,
        fixed_schedule,
        fixed_rate,
        fixed_leg_day_count_convention,
        float_schedule,
        float_index,
        spread,
        float_index.dayCounter()
    )

    return swap, [float_index.fixingDate(x) for x in float_schedule][:-1]


def make_simple_portfolio(list_of_start_dates, list_of_maturity_dates,
                          list_of_nominal_amounts,
                          list_of_float_indices,
                          list_of_fixed_rates,
                          list_of_swap_types):

    simple_portfolio = []

    for (start_date, maturity_date,
         nominal_amount,
         float_index,
         fixed_rate,
         swap_type) in zip(list_of_start_dates, list_of_maturity_dates,
                           list_of_nominal_amounts,
                           list_of_float_indices,
                           list_of_fixed_rates,
                           list_of_swap_types):

        simple_portfolio.append(
            create_plain_vanilla_swap(start_date, maturity_date,
                                      nominal_amount,
                                      float_index,
                                      fixed_rate,
                                      swap_type=swap_type)
        )

    return simple_portfolio


def calculate_portfolio_npv(flat_forward_relinkable_handle, portfolio):

    engine = create_pricing_engine(flat_forward_relinkable_handle)

    portfolio_npv = []
    for deal, _ in portfolio:
        deal.setPricingEngine(engine)
        portfolio_npv.append(deal.NPV())

    return portfolio_npv


# Evaluation grid, curve dates, and NPV matrix...
def define_evaluation_grid(todays_date, simple_portfolio, number_of_months=12*6):

    evaluation_dates_grid = [
        todays_date + ql.Period(i_month, ql.Months) for i_month in range(number_of_months)
        ]

    for deal in simple_portfolio:
        evaluation_dates_grid += deal[1]

    evaluation_dates_grid = np.unique(np.sort(evaluation_dates_grid))

    evaluation_time_grid = np.vectorize(
        lambda x: ql.ActualActual().yearFraction(todays_date, x)
    )(evaluation_dates_grid)

    # diff_evaluation_time_grid = evaluation_time_grid[1:] - evaluation_time_grid[:-1]

    return evaluation_dates_grid, evaluation_time_grid  #, diff_evaluation_time_grid


def define_curve_dates(date, n_years=10):

    # append first half year to date
    curve_dates = [date, date + ql.Period(6, ql.Months)]

    curve_dates += [date + ql.Period(i_year, ql.Years) for i_year in range(1, n_years + 1)]

    return curve_dates


# TODO ...
def create_npv_matrix(todays_date,
                      number_of_paths,
                      evaluation_dates_grid,
                      simple_portfolio,
                      flat_forward,
                      flat_forward_relinkable_handle,
                      zero_bonds,
                      float_index):

    n_dates, n_deals = len(evaluation_dates_grid), len(simple_portfolio)

    npv_matrix = np.zeros(
        (number_of_paths, n_dates, n_deals)
    )

    for i_path in range(number_of_paths):
        for i_date in range(n_dates):

            date = evaluation_dates_grid[i_date]

            discount_curve = get_discount_curve(
                define_curve_dates(date), zero_bonds[i_path, i_date, :]
            )

            set_evaluation_date(date)
            link_to_curve(flat_forward_relinkable_handle, discount_curve)

            # TODO Check... is this correct?
            is_valid_fixing_date = float_index.isValidFixingDate(date)

            if is_valid_fixing_date:
                fixing = float_index.fixing(date)
                float_index.addFixing(date, fixing)

            for i_deal in range(n_deals):
                npv_matrix[i_path, i_date, i_deal] = simple_portfolio[i_deal][0].NPV()

        ql.IndexManager.instance().clearHistories()

    set_evaluation_date(todays_date)
    link_to_curve(flat_forward_relinkable_handle, flat_forward)

    return npv_matrix


def calculate_discounted_npv_matrix(npv_matrix, discount_factors):

    discounted_npv_matrix = np.zeros(npv_matrix.shape)

    for i in range(npv_matrix.shape[2]):
        discounted_npv_matrix[:, :, i] = npv_matrix[:, :, i] * discount_factors

    return discounted_npv_matrix


# Gsr model and simulation of paths...
def generate_gsr_model(flat_forward_handle,
                       volatility_step_dates, volatilities,
                       mean_reversion,
                       forward_measure_time=16.0):

    return ql.Gsr(flat_forward_handle,
                  volatility_step_dates, volatilities,
                  mean_reversion,
                  forward_measure_time)


def generate_paths(number_of_paths,
                   evaluation_time_grid,
                   tenors,
                   inv_cumulative_gaussian_rsg,
                   model):

    n_tenors = len(tenors)

    diff_evaluation_time_grid = evaluation_time_grid[1:] - evaluation_time_grid[:-1]

    x = np.zeros((number_of_paths, len(evaluation_time_grid)))
    y = np.zeros((number_of_paths, len(evaluation_time_grid)))

    zero_bonds = np.zeros(
        (number_of_paths, len(evaluation_time_grid), n_tenors)
    )

    for j_tenor in range(n_tenors):
        zero_bonds[:, 0, j_tenor] = model.zerobond(
            tenors[j_tenor], 0, 0
        )

    process = model.stateProcess()

    for n_path in range(number_of_paths):

        next_sequence = inv_cumulative_gaussian_rsg.nextSequence().value()

        for i_time in range(1, len(evaluation_time_grid)):

            t_start = evaluation_time_grid[i_time - 1]
            t_end = evaluation_time_grid[i_time]

            x[n_path, i_time] = process.expectation(
                t_start, x[n_path, i_time - 1], diff_evaluation_time_grid[i_time - 1]
            ) + next_sequence[i_time-1] * process.stdDeviation(
                t_start, x[n_path, i_time - 1], diff_evaluation_time_grid[i_time - 1]
            )

            # y equals standardized x (see Gsr-paper by Caspers and Gsr model in QuantLib)
            y[n_path, i_time] = \
                (x[n_path, i_time] - process.expectation(0, 0, t_end)) / process.stdDeviation(0, 0, t_end)

            for j_tenor in range(n_tenors):
                zero_bonds[n_path, i_time, j_tenor] = model.zerobond(
                    t_end + tenors[j_tenor], t_end, y[n_path, i_time]
                )

    return x, zero_bonds


# Netting, exposure, and CVA...
def calculate_netted_npv_matrix(npv_matrix):
    return np.sum(npv_matrix, axis=2)


def calculate_exposure(portfolio_npv):

    exposure = portfolio_npv.copy()
    exposure[exposure < 0] = 0

    return exposure


def calculate_expected_exposure(portfolio_npv, number_of_paths):
    return np.sum(
        calculate_exposure(portfolio_npv), axis=0
    ) / number_of_paths


def calculate_potential_future_exposure(exposure, number_of_paths, quantile=0.95):

    # TODO Should we have ceil here?
    potential_future_exposure = np.apply_along_axis(
        lambda x: np.sort(x)[int(np.ceil(quantile * number_of_paths))], 0, exposure
    )

    # Alternative formulation: use max of each exposure path
    # potential_future_exposure = np.sort(np.max(exposure, axis=1))[quantile * number_of_paths]

    return potential_future_exposure


def calculate_economic_cva(expected_discounted_exposure, default_probabilities, recovery_rate=0.4):
    return (1 - recovery_rate) * np.sum(
        expected_discounted_exposure[1:] * default_probabilities
    )


# Plotting functions...

# def plot_npv_paths(n_first, n_last,
#                    evaluation_time_grid,
#                    portfolio_npv, discounted_portfolio_npv):
#
#     _, (axis_1, axis_2) = plt.subplots(2, 1, figsize=(12, 10), sharey=True)
#
#     for i_path in range(n_first, n_last):
#         axis_1.plot(evaluation_time_grid, portfolio_npv[i_path, :])
#
#     axis_1.set_xlabel("Years")
#     axis_1.set_ylabel("Portfolio NPV")
#     axis_1.set_title("Portfolio NPV paths")
#
#     for i_path in range(n_first, n_last):
#         axis_2.plot(evaluation_time_grid, discounted_portfolio_npv[i_path, :])
#
#     axis_2.set_xlabel("Years")
#     axis_2.set_ylabel("Discounted Portfolio NPV")
#     axis_2.set_title("Discounted portfolio NPV paths")
#
#
# def plot_exposure_paths(n_first, n_last,
#                         evaluation_time_grid,
#                         exposure, discounted_exposure):
#
#     _, (axis_1, axis_2) = plt.subplots(2, 1, figsize=(12, 10))  # , sharey=True)
#
#     for i_path in range(n_first, n_last):
#         axis_1.plot(evaluation_time_grid, exposure[i_path, :])
#
#     axis_1.set_ylim([-10000, 70000])
#     axis_1.set_xlabel("Years")
#     axis_1.set_ylabel("Exposure")
#     axis_1.set_title("Exposure paths")
#
#     for i_path in range(n_first, n_last):
#         axis_2.plot(evaluation_time_grid, discounted_exposure[i_path, :])
#
#     axis_2.set_ylim([-10000, 70000])
#     axis_2.set_xlabel("Years")
#     axis_2.set_ylabel("Discounted Exposure")
#     axis_2.set_title("Discounted exposure paths")
#
#
# def plot_expected_exposure_paths(evaluation_time_grid,
#                                  expected_exposure, expected_discounted_exposure):
#
#     _, (axis_1, axis_2) = plt.subplots(2, 1, figsize=(8, 10))  # , sharey=True)
#
#     axis_1.plot(evaluation_time_grid, expected_exposure)
#
#     axis_1.set_xlabel("Time in years")
#     axis_1.set_ylabel("Exposure")
#     axis_1.set_title("Expected exposure")
#
#     axis_2.plot(evaluation_time_grid, expected_discounted_exposure)
#
#     axis_2.set_xlabel("Time in years")
#     axis_2.set_ylabel("Discounted Exposure")
#     axis_2.set_title("Expected discounted exposure")
#
#
# def plot_expected_discounted_exposure(evaluation_time_grid,
#                                       expected_discounted_exposure):
#
#     # plt.figure(figsize=(7, 5), dpi=300)
#     plt.figure()
#     plt.plot(evaluation_time_grid, expected_discounted_exposure)
#
#     plt.ylim([-2000, 10000])
#     plt.xlabel("Years")
#     plt.ylabel("Expected discounted exposure")
#     plt.title("Expected discounted exposure")
#
#
# def plot_potential_future_exposure(evaluation_time_grid,
#                                    potential_future_exposure):
#
#     # plt.figure(figsize=(7, 5), dpi=300)
#     plt.figure()
#     plt.plot(evaluation_time_grid, potential_future_exposure)
#
#     plt.xlabel("Years")
#     plt.ylabel("Potential future exposure")
#     plt.ylim([-2000, 35000])
#
#     plt.title("Potential future exposure")
#
#
# def plot_default_curve(times, default_curve):
#
#     _, ((axis_1, axis_2), (axis_3, axis_4)) = plt.subplots(2, 2, figsize=(10, 10))
#
#     default_probability = get_default_probability(times, default_curve)
#
#     axis_1.plot(times, default_probability)
#
#     axis_1.set_xlabel("Years")
#     axis_1.set_ylabel("Probability")
#     axis_1.set_title("Default probability")
#
#     survival_probability = get_survival_probability(times, default_curve)
#
#     axis_2.plot(times, survival_probability)
#
#     axis_2.set_xlabel("Years")
#     axis_2.set_ylabel("Probability")
#     axis_2.set_title("Survival probability")
#
#     default_density = get_default_density(times, default_curve)
#
#     axis_3.plot(times, default_density)
#
#     axis_3.set_xlabel("Years")
#     axis_3.set_ylabel("Density")
#     axis_3.set_title("Default density")
#
#     hazard_rate = get_hazard_rate(times, default_curve)
#
#     axis_4.plot(times, hazard_rate)
#
#     axis_4.set_xlabel("Years")
#     axis_4.set_ylabel("Rate")
#     axis_4.set_title("Hazard rate")
#
#
def main():

    # Set evaluation date...
    # todays_date = ql.Date(7, 4, 2015)
    todays_date = ql.Date(13, 8, 2015)
    # ql.Settings.instance().setEvaluationDate(todays_date)
    set_evaluation_date(todays_date)

    # Market data...
    rate = ql.SimpleQuote(0.03)

    flat_forward, flat_forward_handle, flat_forward_relinkable_handle = \
        create_flat_forward(todays_date, rate)

    Euribor6M = ql.Euribor6M(flat_forward_relinkable_handle)

    # Create simple swap portfolio...
    list_of_start_dates = [
        todays_date + ql.Period("2d"),
        todays_date + ql.Period("2d")
    ]

    list_of_maturity_dates = [ql.Period(years) for years in ["5Y", "4Y"]]

    list_of_nominal_amounts = [1E6, 5E5]
    list_of_float_indices = [Euribor6M, Euribor6M]
    list_of_fixed_rates = [0.03, 0.03]

    list_of_swap_types = [ql.VanillaSwap.Payer, ql.VanillaSwap.Receiver]

    simple_portfolio = make_simple_portfolio(
        list_of_start_dates, list_of_maturity_dates,
        list_of_nominal_amounts,
        list_of_float_indices,
        list_of_fixed_rates,
        list_of_swap_types
    )

    portfolio_npv = calculate_portfolio_npv(flat_forward_relinkable_handle, simple_portfolio)

    # Instantiate the Gsr model...

    volatility_step_dates = [todays_date + 100]

    volatilities = [
        ql.QuoteHandle(ql.SimpleQuote(0.0075)),
        ql.QuoteHandle(ql.SimpleQuote(0.0075))
    ]

    mean_reversion = [ql.QuoteHandle(ql.SimpleQuote(0.02))]

    gsr_model = generate_gsr_model(flat_forward_handle,
                                   volatility_step_dates, volatilities,
                                   mean_reversion,
                                   forward_measure_time=16.0)

    # Create evaluation grid and simulate paths (using the Gsr model)...

    evaluation_dates_grid, evaluation_time_grid = \
        define_evaluation_grid(todays_date, simple_portfolio)

    inv_cumulative_gaussian_rsg = create_random_number_generator(evaluation_time_grid)

    number_of_paths = 1500
    tenors = np.array([0.0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    x, zero_bonds = generate_paths(
        number_of_paths, evaluation_time_grid, tenors, inv_cumulative_gaussian_rsg, gsr_model
    )

    # Plot paths...
    # for i in range(number_of_paths):
    #     plt.plot(evaluation_time_grid, x[i, :])

    # Create the discounted NPV matrix...
    npv_matrix = create_npv_matrix(
        todays_date,
        number_of_paths,
        evaluation_dates_grid,
        simple_portfolio,
        flat_forward,
        flat_forward_relinkable_handle,
        zero_bonds,
        Euribor6M
    )

    discount_factors = generate_discount_factors(flat_forward_handle, evaluation_time_grid)

    discounted_npv_cube = calculate_discounted_npv_matrix(npv_matrix, discount_factors)

    # Calculate the portfolio NPV for the netting set...
    portfolio_npv = calculate_netted_npv_matrix(npv_matrix)
    discounted_portfolio_npv = calculate_netted_npv_matrix(discounted_npv_cube)

#     # Plot the first NPV paths...
#     n_first, n_last = 0, 30
#     plot_npv_paths(n_first, n_last,
#                    evaluation_time_grid,
#                    portfolio_npv, discounted_portfolio_npv)

    # Calculate the exposure and discounted exposure...
    exposure = calculate_exposure(portfolio_npv)
    discounted_exposure = calculate_exposure(discounted_portfolio_npv)

#     # Plot the first exposure paths...
#     n_first, n_last = 0, 30
#     plot_exposure_paths(n_first, n_last,
#                         evaluation_time_grid,
#                         exposure, discounted_exposure)
#
    # Calculate the "expected" and the "expected discounted" exposure...
    expected_exposure = calculate_expected_exposure(portfolio_npv, number_of_paths)
    expected_discounted_exposure = calculate_expected_exposure(discounted_portfolio_npv, number_of_paths)

#     # Plot the "expected" and the "expected discounted" exposure paths...
#     plot_expected_exposure_paths(evaluation_time_grid,
#                                  expected_exposure, expected_discounted_exposure)
#
#     plot_expected_discounted_exposure(evaluation_time_grid,
#                                       expected_discounted_exposure)
#
    # Calculate the PFE (corresponding to the default 95% quantile)...
    potential_future_exposure = \
        calculate_potential_future_exposure(exposure, number_of_paths)

#     plot_potential_future_exposure(evaluation_time_grid,
#                                    potential_future_exposure)

    # calculate the maximum PFE...
    max_potential_future_exposure = np.max(potential_future_exposure)

    # Default curve
    default_dates = [todays_date + ql.Period(i_year, ql.Years) for i_year in range(11)]
    hazard_rates = [0.02 * i_year for i_year in range(11)]

    default_curve = create_default_curve(default_dates, hazard_rates)

#     # Plot default curves (default and survival probabilities, default densities, and hazard rates)...
#
#     default_times = np.linspace(0, 30, 100)
#     plot_default_curve(default_times, default_curve)

    # Calculate default probabilities...
    default_probabilities = \
        calculate_default_probability_grid(evaluation_time_grid, default_curve)

    # Calculation of the CVA...
    economic_cva = calculate_economic_cva(expected_discounted_exposure, default_probabilities, recovery_rate=0.4)
    print(economic_cva)

    # List of TODOs...

    # TODO Use QuantLib to calculate CCR and CVA REA, and KVA with SA-CCR
    # TODO Add doc tests to functions
    # TODO Use pandas to request data and SQLite or MySQL as data repositories


if __name__ == '__main__':
    main()
